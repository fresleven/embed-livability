"""
This code is adapted from the image.py by Kiela et al. (2020) in https://github.com/facebookresearch/mmbt/blob/master/mmbt/models/image.py
and the equivalent Huggingface implementation: utils_mmimdb.py, which can be
found here: https://github.com/huggingface/transformers/blob/8ea412a86faa8e9edeeb6b5c46b08def06aa03ea/examples/research_projects/mm-imdb/utils_mmimdb.py

The ImageEncoderDenseNet class is modified from the original ImageEncoder class to be based on pre-trained DenseNet
instead of ResNet and to be able to load saved pre-trained weights.

This class makes up the image submodule of the MMBT model.

The forward function is also modified according to the forward function of the DenseNet model listed here:

Original forward function of DenseNet

def forward(self, x):
    features = self.features(x)
    out = F.relu(features, inplace=True)
    out = F.adaptive_avg_pool2d(out, (1, 1))
    out = torch.flatten(out, 1)
    out = self.classifier(out)
    return out
"""
import os
import logging
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

# Import CVAE utilities
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from cvae_utils import reparameterize

logger = logging.getLogger(__name__)

# mapping number of image embeddings to AdaptiveAvgPool2d output size
POOLING_BREAKDOWN = {1: (1, 1), 2: (2, 1), 3: (3, 1), 4: (2, 2), 5: (5, 1), 6: (3, 2), 7: (7, 1), 8: (4, 2), 9: (3, 3)}

# module assumes that the directory where the saved chexnet weight is in the same level as this module
MMBT_DIR_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Data root: defaults to <repo>/data_livability, override with $LIVABILITY_DATA_DIR
DATA_DIR = os.environ.get("LIVABILITY_DATA_DIR", os.path.join(MMBT_DIR_PARENT, "data_livability"))
MODELS_DIR = os.path.join(DATA_DIR, "models")
SAVED_CHEXNET = os.path.join(MODELS_DIR, "saved_chexnet.pt")


class CVAEEncoder(nn.Module):
    """
    Encoder for Conditional VAE to learn latent confounder Z from treatments and covariates.
    Takes treatments A_s, A_Ns and covariates X_s, X_Ns (image and text) as input.
    Downsamples to target latent_hw and projects to mu/logvar (no upsampling back).
    Uses convolutional layers instead of FC to maintain spatial structure.
    """
    def __init__(self, treatment_channels=3, covariate_channels=3, latent_hw=50, 
                 latent_d=32, base_channels=64, depth=3, dropout=0.1, text_embedding_dim=None):
        super().__init__()
        self.latent_h = latent_hw
        self.latent_w = latent_hw
        self.latent_d = latent_d
        self.treatment_channels = treatment_channels
        self.depth = depth
        self.text_embedding_dim = text_embedding_dim
        
        # Encoder: input channels = treatment_channels + covariate_channels (after resampling)
        # Text embeddings will be projected and added spatially
        input_channels = treatment_channels + covariate_channels
        
        # Project text embeddings to spatial feature map if provided
        if text_embedding_dim is not None:
            self.text_proj = nn.Linear(text_embedding_dim, base_channels)
            input_channels += base_channels
        
        # Downsampling path (stride-2) to progressively reduce spatial dims
        downsample_layers = [
            nn.Conv2d(input_channels, base_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
        ]
        
        in_channels = base_channels
        for _ in range(depth - 1):
            out_channels = min(in_channels * 2, 512)  # Cap at 512
            downsample_layers += [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Dropout2d(dropout),
            ]
            in_channels = out_channels
        
        self.downsample = nn.Sequential(*downsample_layers)
        self.final_channels = in_channels
        
        # Use 1x1 convolutions to project to mu and logvar (no FC layers)
        self.conv_mu = nn.Conv2d(in_channels, latent_d, kernel_size=1)
        self.conv_logvar = nn.Conv2d(in_channels, latent_d, kernel_size=1)
        
        # Adaptive pooling to ensure output is exactly latent_hw
        self.pool_to_latent = nn.AdaptiveAvgPool2d((latent_hw, latent_hw))
    
    def forward(self, treatments, covariates, text_embeddings=None):
        """
        Args:
            treatments: (B, C_A, H_A, W_A) - treatments (can have any spatial dims)
            covariates: (B, C_X, H_X, W_X) - covariates (can have different spatial dims)
            text_embeddings: (B, text_embedding_dim) - optional text embeddings to condition on
        Returns:
            mu, logvar: (B, latent_d, latent_h, latent_w) - parameters of posterior distribution
        """
        # Resample covariates to match treatment spatial dimensions
        treatment_h, treatment_w = treatments.shape[2], treatments.shape[3]
        if covariates.shape[2] != treatment_h or covariates.shape[3] != treatment_w:
            covariates = F.adaptive_avg_pool2d(covariates, (treatment_h, treatment_w))
        
        # Concatenate treatments and covariates
        x = torch.cat([treatments, covariates], dim=1)  # (B, C_A + C_X, H, W)
        
        # Add text embeddings if provided
        if text_embeddings is not None and self.text_embedding_dim is not None:
            # Project text embeddings: (B, text_dim) -> (B, base_channels)
            text_features = self.text_proj(text_embeddings)  # (B, base_channels)
            # Expand to spatial dimensions and concatenate
            text_spatial = text_features.unsqueeze(-1).unsqueeze(-1)  # (B, base_channels, 1, 1)
            text_spatial = text_spatial.expand(-1, -1, treatment_h, treatment_w)  # (B, base_channels, H, W)
            x = torch.cat([x, text_spatial], dim=1)  # (B, C_A + C_X + base_channels, H, W)
        
        # Downsample through convolutional layers
        encoded = self.downsample(x)  # (B, final_channels, h', w')
        
        # Project to mu and logvar
        mu = self.conv_mu(encoded)  # (B, latent_d, h', w')
        logvar = self.conv_logvar(encoded)  # (B, latent_d, h', w')
        
        # Pool to exact target latent_hw dimensions
        mu = self.pool_to_latent(mu)  # (B, latent_d, latent_h, latent_w)
        logvar = self.pool_to_latent(logvar)  # (B, latent_d, latent_h, latent_w)
        
        return mu, logvar

class CVAEDecoder(nn.Module):
    """
    Decoder for Conditional VAE to reconstruct treatments from latent confounder and covariates.
    Outputs distribution parameters for zero-inflated distributions:
    - RS: Zero-Inflated Normal (3 params: logit_p, mean, logvar)
    - DSM: Zero-Inflated Gamma (3 params: logit_p, log_shape, log_scale)
    - GIU_RGB: Zero-Inflated Log-Normal (3 params: logit_p, mean, logvar)
    """
    def __init__(self, treatment_channels=3, treatment_hw=50, covariate_channels=3, latent_hw=50,
                 latent_d=32, base_channels=64, depth=3, dropout=0.1, cvae_treatment_source="alphaearth", 
                 var=True, text_embedding_dim=None):
        super().__init__()
        self.latent_h = latent_hw
        self.latent_w = latent_hw
        self.latent_d = latent_d
        self.treatment_hw = treatment_hw
        self.output_hw = treatment_hw // 5  # Output is 1/5 of treatment_hw
        self.cvae_treatment_source = cvae_treatment_source
        self.text_embedding_dim = text_embedding_dim
        self.var = var
        
        # Pool covariates from arbitrary dimensions down to latent spatial resolution
        self.cov_pool = nn.AdaptiveAvgPool2d((latent_hw, latent_hw))
        
        # Project latent and covariates separately using 1x1 convolutions
        self.latent_proj = nn.Conv2d(latent_d, base_channels, kernel_size=1)
        self.cov_proj = nn.Conv2d(covariate_channels, base_channels, kernel_size=1)
        
        # Project text embeddings if provided
        if text_embedding_dim is not None:
            self.text_proj = nn.Linear(text_embedding_dim, base_channels)
        
        # Decoder: refine features progressively with stride-1 convolutions
        layers = []
        in_channels = 2 * base_channels
        
        for i in range(depth):
            out_channels = max(base_channels, in_channels // 2) if i < depth - 1 else base_channels
            
            layers += [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            ]
            
            if i < depth - 1:
                layers += [
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Dropout2d(dropout),
                ]
            
            in_channels = out_channels
        
        self.decoder = nn.Sequential(*layers)

        # Adaptive pooling to get output to 1/5 of treatment_hw
        self.output_pool = nn.AdaptiveAvgPool2d((self.output_hw, self.output_hw))

        # --- Parameter heads for each dataset ---
        
        # RS: Zero-Inflated Normal (1 channel each)
        self.rs_mean = nn.Conv2d(base_channels, 1, kernel_size=3, padding=1)
        if self.var:
            self.rs_logit_p = nn.Conv2d(base_channels, 1, kernel_size=3, padding=1)
            self.rs_logvar = nn.Conv2d(base_channels, 1, kernel_size=3, padding=1)
        
        # DSM: Zero-Inflated Gamma (1 channel each)
        self.dsm_mean = nn.Conv2d(base_channels, 1, kernel_size=3, padding=1)
        if self.var:
            self.dsm_logit_p = nn.Conv2d(base_channels, 1, kernel_size=3, padding=1)
            self.dsm_log_shape = nn.Conv2d(base_channels, 1, kernel_size=3, padding=1)
            self.dsm_log_scale = nn.Conv2d(base_channels, 1, kernel_size=3, padding=1)
        
        # GIU_RGB: Zero-Inflated Log-Normal (1 channel each)
        self.giu_mean = nn.Conv2d(base_channels, 1, kernel_size=3, padding=1)
        if self.var:
            self.giu_logit_p = nn.Conv2d(base_channels, 1, kernel_size=3, padding=1)
            self.giu_logvar = nn.Conv2d(base_channels, 1, kernel_size=3, padding=1)
    
    def forward(self, z, covariates, text_embeddings=None, normalize_output=True):
        """
        Args:
            z: (B, latent_d, latent_h, latent_w) - latent representation
            covariates: (B, C_X, H, W) - covariates at arbitrary spatial dims (e.g., 224x224)
            text_embeddings: (B, text_embedding_dim) - optional text embeddings to condition on
            normalize_output: bool - not used in this version (kept for compatibility)
        Returns:
            mean: (B, 3, output_hw, output_hw) - concatenated means [RS, DSM, GIU]
            kappa: dict with additional parameters for each distribution
                - 'rs': {'logit_p': (B,1,H,W), 'logvar': (B,1,H,W)}
                - 'dsm': {'logit_p': (B,1,H,W), 'log_shape': (B,1,H,W), 'log_scale': (B,1,H,W)}
                - 'giu': {'logit_p': (B,1,H,W), 'logvar': (B,1,H,W)}
        """
        # Pool covariates to latent spatial resolution
        cov_pooled = self.cov_pool(covariates)  # (B, C_X, latent_h, latent_w)
        
        # Project latent and covariates using 1x1 convolutions
        latent_proj = self.latent_proj(z)  # (B, base_channels, latent_h, latent_w)
        cov_proj = self.cov_proj(cov_pooled)  # (B, base_channels, latent_h, latent_w)
        
        # Concatenate projected features
        x = torch.cat([latent_proj, cov_proj], dim=1)  # (B, 2*base_channels, latent_h, latent_w)
        
        # Add text embeddings if provided
        if text_embeddings is not None and self.text_embedding_dim is not None:
            text_features = self.text_proj(text_embeddings)  # (B, base_channels)
            text_spatial = text_features.unsqueeze(-1).unsqueeze(-1)  # (B, base_channels, 1, 1)
            text_spatial = text_spatial.expand(-1, -1, x.shape[2], x.shape[3])  # (B, base_channels, latent_h, latent_w)
            x = torch.cat([x, text_spatial], dim=1)  # (B, 3*base_channels, latent_h, latent_w)
        
        # Decode with stride-1 convolutions
        x = self.decoder(x)  # (B, base_channels, latent_h, latent_w)
        
        # Pool down to 1/5 of treatment_hw
        x = self.output_pool(x)  # (B, base_channels, output_hw, output_hw)
        
        # --- Generate parameters for each distribution ---
        
        # RS: Zero-Inflated Normal
        rs_mean = self.rs_mean(x)  # (B, 1, H, W)
        
        # DSM: Zero-Inflated Gamma
        dsm_mean = self.dsm_mean(x)  # (B, 1, H, W)
        
        # GIU: Zero-Inflated Log-Normal
        giu_mean = self.giu_mean(x)  # (B, 1, H, W)
        
        # Concatenate means: [RS, DSM, GIU]
        mean = torch.cat([rs_mean, dsm_mean, giu_mean], dim=1)  # (B, 3, H, W)
        
        # Additional parameters (kappa/variance parameters)
        kappa = None
        if self.var:
            kappa = {
                'rs': {
                    'logit_p': self.rs_logit_p(x),  # (B, 1, H, W)
                    'logvar': self.rs_logvar(x)     # (B, 1, H, W)
                },
                'dsm': {
                    'logit_p': self.dsm_logit_p(x),     # (B, 1, H, W)
                    'log_shape': self.dsm_log_shape(x),  # (B, 1, H, W)
                    'log_scale': self.dsm_log_scale(x)   # (B, 1, H, W)
                },
                'giu': {
                    'logit_p': self.giu_logit_p(x),  # (B, 1, H, W)
                    'logvar': self.giu_logvar(x)     # (B, 1, H, W)
                }
            }

        return mean, kappa

# class CVAEDecoder(nn.Module):
#     """
#     Decoder for Conditional VAE to reconstruct treatments from latent confounder and covariates.
#     Covariates come in at arbitrary dimensions (e.g., 224x224) and are pooled to latent spatial resolution.
#     Uses only convolutional layers (no FC). Can optionally condition on text embeddings.
#     """
#     def __init__(self, treatment_channels=3, treatment_hw=50, covariate_channels=3, latent_hw=50,
#                  latent_d=32, base_channels=64, depth=3, dropout=0.1, cvae_treatment_source="alphaearth", var=True, text_embedding_dim=None):
#         super().__init__()
#         self.latent_h = latent_hw
#         self.latent_w = latent_hw
#         self.latent_d = latent_d
#         self.treatment_hw = treatment_hw
#         self.output_hw = treatment_hw // 5  # Output is 1/5 of treatment_hw
#         self.cvae_treatment_source = cvae_treatment_source
#         self.text_embedding_dim = text_embedding_dim
        
#         # Pool covariates from arbitrary dimensions down to latent spatial resolution
#         self.cov_pool = nn.AdaptiveAvgPool2d((latent_hw, latent_hw))
        
#         # Project latent and covariates separately using 1x1 convolutions
#         self.latent_proj = nn.Conv2d(latent_d, base_channels, kernel_size=1)
#         self.cov_proj = nn.Conv2d(covariate_channels, base_channels, kernel_size=1)
        
#         # Project text embeddings if provided
#         if text_embedding_dim is not None:
#             self.text_proj = nn.Linear(text_embedding_dim, base_channels)
        
#         # Decoder: refine features progressively with stride-1 convolutions
#         # Maintains spatial dims (latent_hw == treatment_hw throughout)
#         layers = []
#         in_channels = 2 * base_channels
        
#         for i in range(depth):
#             out_channels = max(base_channels, in_channels // 2) if i < depth - 1 else treatment_channels
            
#             layers += [
#                 nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
#             ]
            
#             if i < depth - 1:
#                 layers += [
#                     nn.BatchNorm2d(out_channels),
#                     nn.ReLU(inplace=True),
#                     nn.Dropout2d(dropout),
#                 ]
#             # else:
#             #     layers.append(nn.Sigmoid())  # Output between 0 and 1 for image-like outputs
            
#             in_channels = out_channels
        
#         self.decoder = nn.Sequential(*layers)

#         # Adaptive pooling to get output to 1/5 of treatment_hw
#         self.output_pool = nn.AdaptiveAvgPool2d((self.output_hw, self.output_hw))

#         # VMF parameters: mean direction and concentration
#         self.var = var
#         self.final_conv_mean = nn.Conv2d(in_channels, treatment_channels, kernel_size=3, padding=1)
#         C = 1 if self.treatment_hw == 50 else treatment_channels # kappa for VMF is 1 per pixel else treatment_channels per pixel for Gaussian treatments
#         if self.var:
#             self.final_conv_kappa = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)
    
#     def forward(self, z, covariates, text_embeddings=None, normalize_output=True):
#         """
#         Args:
#             z: (B, latent_d, latent_h, latent_w) - latent representation
#             covariates: (B, C_X, H, W) - covariates at arbitrary spatial dims (e.g., 224x224)
#             text_embeddings: (B, text_embedding_dim) - optional text embeddings to condition on
#             normalize_output: bool - if True, normalize mean directions to unit norm per pixel
#         Returns:
#             mean_direction: (B, C_A, treatment_hw, treatment_hw) - mean direction (normalized)
#             kappa: (B, 1, treatment_hw, treatment_hw) - concentration parameter (positive)
#         """
#         # Pool covariates to latent spatial resolution
#         cov_pooled = self.cov_pool(covariates)  # (B, C_X, latent_h, latent_w)
        
#         # Project latent and covariates using 1x1 convolutions
#         latent_proj = self.latent_proj(z)  # (B, base_channels, latent_h, latent_w)
#         cov_proj = self.cov_proj(cov_pooled)  # (B, base_channels, latent_h, latent_w)
        
#         # Concatenate projected features
#         x = torch.cat([latent_proj, cov_proj], dim=1)  # (B, 2*base_channels, latent_h, latent_w)
        
#         # Add text embeddings if provided
#         if text_embeddings is not None and self.text_embedding_dim is not None:
#             text_features = self.text_proj(text_embeddings)  # (B, base_channels)
#             text_spatial = text_features.unsqueeze(-1).unsqueeze(-1)  # (B, base_channels, 1, 1)
#             text_spatial = text_spatial.expand(-1, -1, x.shape[2], x.shape[3])  # (B, base_channels, latent_h, latent_w)
#             x = torch.cat([x, text_spatial], dim=1)  # (B, 3*base_channels, latent_h, latent_w)
        
#         # Decode with stride-1 convolutions (spatial dims remain constant)
#         x = self.decoder(x)  # (B, C_A, latent_h, latent_w) == (B, C_A, treatment_hw, treatment_hw)
        
#         # Pool down to 1/5 of treatment_hw
#         x = self.output_pool(x)  # (B, C_A, output_hw, output_hw)
        
#         mean = self.final_conv_mean(x)
#         kappa = None
#         if self.var:
#             kappa = self.final_conv_kappa(x)

#         if self.cvae_treatment_source == "alphaearth":
#             if normalize_output:
#                 mean = F.normalize(mean, dim=1, eps=1e-8)
            
#             # Ensure kappa is positive (use softplus)
#             if self.var:
#                 kappa = F.softplus(kappa)  # (B, 1, H, W)

#         return mean, kappa


class ImageEncoderDenseNet(nn.Module):
    def __init__(self, num_image_embeds, saved_model=True, path=os.path.join(MODELS_DIR, SAVED_CHEXNET),
        modal_hidden_size=1024,
        alphaearth=False,
        num_image_embeds_ae=3,
        ae_base_channels=128,
        ae_depth=3,
        ae_dropout=0.1,
        anysat=False,
        num_image_embeds_anysat=3,
        anysat_base_channels=128,
        anysat_depth=3,
        anysat_dropout=0.1,
        terramind=False,
        num_image_embeds_terramind=3,
        terramind_base_channels=128,
        terramind_depth=3,
        terramind_dropout=0.1,
        cvae=False,
        num_image_embeds_cvae=3,
        cvae_base_channels=128,
        cvae_depth=3,
        cvae_dropout=0.1,
        cvae_latent_d=32,
        cvae_encoder_base_channels=64,
        cvae_encoder_depth=3,
        cvae_decoder_base_channels=64,
        cvae_decoder_depth=3,
        cvae_kl_weight=1.0,
        cvae_treatment_source="alphaearth",
        cvae_covariate_source="image",
        var=True,
        cvae_latent_hw=50,
        text_embedding_dim=None,
        ):
        """

        :type num_image_embeds: int
        :param num_image_embeds: number of image embeddings to generate; 1-9 as they map to specific numbers of pooling
        output shape in the 'POOLING_BREAKDOWN'
        :param saved_model: True to load saved pre-trained model False to use torch pre-trained model
        :param path: path to the saved .pt model file
        :param cvae_*: C-VAE hyperparameters for confounder reconstruction
        """
        super().__init__()
        if saved_model and os.path.exists(path):
            # loading pre-trained weight, e.g. ChexNet
            # the model here expects the weight to be regular Tensors and NOT cuda Tensor
            model = torch.load(path, weights_only=False)
            logger.info(f"Saved model loaded from: {path}")
        else:
            if saved_model:
                logger.warning(
                    f"Pretrained DenseNet weights not found at {path}; falling back to "
                    f"torchvision ImageNet DenseNet-121. Set $LIVABILITY_DATA_DIR/models/"
                    f"saved_chexnet.pt to reproduce the paper's exact feature extractor."
                )
            model = torchvision.models.densenet121(pretrained=True)

        # DenseNet architecture last layer is the classifier; we only want everything before that
        modules = list(model.children())[:-1]
        self.model = nn.Sequential(*modules)
        # self.model same as original DenseNet self.features part of the forward function
        self.pool = nn.AdaptiveAvgPool2d(POOLING_BREAKDOWN[3])
       # self.pool = nn.AdaptiveAvgPool2d((3,1))
        self.cvae = cvae
        self.cvae_kl_weight = cvae_kl_weight
        self.cvae_treatment_source = cvae_treatment_source
        self.cvae_covariate_source = cvae_covariate_source
        self.alphaearth_flag = alphaearth
        self.anysat_flag = anysat
        self.terramind_flag = terramind

        if alphaearth:
            C = ae_base_channels
            layers = [
                nn.Conv2d(64, C, kernel_size=3, padding=1),
                nn.BatchNorm2d(C),
                nn.ReLU(inplace=True),
            ]
            for _ in range(ae_depth - 1):
                layers += [
                    nn.Conv2d(C, C, kernel_size=3, padding=1),
                    nn.BatchNorm2d(C),
                    nn.ReLU(inplace=True),
                ]
            self.ae_conv = nn.Sequential(*layers)
            self.ae_pool = nn.AdaptiveAvgPool2d(POOLING_BREAKDOWN[num_image_embeds_ae])
            self.ae_proj = nn.Sequential(
                nn.Linear(C, modal_hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(ae_dropout),
            )

        if anysat:
            C = anysat_base_channels
            layers = [
                nn.Conv2d(1536, C, kernel_size=3, padding=1),
                nn.BatchNorm2d(C),
                nn.ReLU(inplace=True),
            ]
            for _ in range(anysat_depth - 1):
                layers += [
                    nn.Conv2d(C, C, kernel_size=3, padding=1),
                    nn.BatchNorm2d(C),
                    nn.ReLU(inplace=True),
                ]
            self.anysat_conv = nn.Sequential(*layers)
            self.anysat_pool = nn.AdaptiveAvgPool2d(POOLING_BREAKDOWN[num_image_embeds_anysat])
            self.anysat_proj = nn.Sequential(
                nn.Linear(C, modal_hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(anysat_dropout),
            )

        if terramind:
            C = terramind_base_channels
            layers = [
                nn.Conv2d(384, C, kernel_size=3, padding=1),
                nn.BatchNorm2d(C),
                nn.ReLU(inplace=True),
            ]
            for _ in range(terramind_depth - 1):
                layers += [
                    nn.Conv2d(C, C, kernel_size=3, padding=1),
                    nn.BatchNorm2d(C),
                    nn.ReLU(inplace=True),
                ]
            self.terramind_conv = nn.Sequential(*layers)
            self.terramind_pool = nn.AdaptiveAvgPool2d(POOLING_BREAKDOWN[num_image_embeds_terramind])
            self.terramind_proj = nn.Sequential(
                nn.Linear(C, modal_hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(terramind_dropout),
            )

        if cvae:
            treatment_channels = 64 if cvae_treatment_source == "alphaearth" else 9
            covariate_channels = 64 if cvae_covariate_source == "alphaearth" else 9
            # cvae_latent_hw = 50 if cvae_treatment_source == "alphaearth" else 224
            cvae_latent_hw = cvae_latent_hw
            # Initialize CVAE encoder and decoder
            self.cvae_encoder = CVAEEncoder(
                treatment_channels=treatment_channels,
                covariate_channels=covariate_channels,
                latent_hw=cvae_latent_hw,
                latent_d=cvae_latent_d,
                base_channels=cvae_encoder_base_channels,
                depth=cvae_encoder_depth,
                dropout=cvae_dropout,
                text_embedding_dim=text_embedding_dim
            )
            
            self.cvae_decoder = CVAEDecoder(
                treatment_channels=treatment_channels,
                treatment_hw=50 if cvae_treatment_source == "alphaearth" else 224,
                covariate_channels=covariate_channels,
                latent_hw=cvae_latent_hw,
                latent_d=cvae_latent_d,
                base_channels=cvae_decoder_base_channels,
                depth=cvae_decoder_depth,
                dropout=cvae_dropout,
                cvae_treatment_source=cvae_treatment_source,
                var=var,
                text_embedding_dim=text_embedding_dim
            )
            
            # For embedding the latent variable
            C = cvae_base_channels
            layers = [
                nn.Conv2d(cvae_latent_d, C, kernel_size=3, padding=1),
                nn.BatchNorm2d(C),
                nn.ReLU(inplace=True),
            ]
            for _ in range(cvae_depth - 1):
                layers += [
                    nn.Conv2d(C, C, kernel_size=3, padding=1),
                    nn.BatchNorm2d(C),
                    nn.ReLU(inplace=True),
                ]
            self.cvae_conv = nn.Sequential(*layers)
            self.cvae_pool = nn.AdaptiveAvgPool2d(POOLING_BREAKDOWN[num_image_embeds_cvae])
            self.cvae_proj = nn.Sequential(
                nn.Linear(C, modal_hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(cvae_dropout),
            )

    def _dense_head(self, features):
        # features: B x 1024 x 7 x 7 (DenseNet-121)
        out = F.relu(features, inplace=True)
        out = self.pool(out)                 # B x 1024 x h x w, h*w = N
        out = torch.flatten(out, start_dim=2)  # B x 1024 x N
        out = out.transpose(1, 2).contiguous() # B x N x 1024
        return out

    def _ae_head(self, tensor_ae):
        # tensor_ae: B x 64 x 50 x 50
        feats = self.ae_conv(tensor_ae)
        feats = self.ae_pool(feats)
        feats = feats.flatten(2)
        feats = feats.transpose(1, 2).contiguous()
        feats = self.ae_proj(feats)             # B x N_ae x 1024
        return feats

    def _anysat_head(self, tensor_anysat):
        # tensor_anysat: B x 1536 x 24 x 24
        feats = self.anysat_conv(tensor_anysat)
        feats = self.anysat_pool(feats)
        feats = feats.flatten(2)
        feats = feats.transpose(1, 2).contiguous()
        feats = self.anysat_proj(feats)         # B x N_anysat x 1024
        return feats

    def _terramind_head(self, tensor_terramind):
        # tensor_terramind: B x 384 x 14 x 14
        feats = self.terramind_conv(tensor_terramind)
        feats = self.terramind_pool(feats)
        feats = feats.flatten(2)
        feats = feats.transpose(1, 2).contiguous()
        feats = self.terramind_proj(feats)      # B x N_terramind x 1024
        return feats

    def _cvae_head(self, z):
        # z: B x latent_d x latent_h x latent_w        
        feats = self.cvae_conv(z)                    # B x C x H x W
        feats = self.cvae_pool(feats)               # B x C x h x w, h*w = N_cvae
        feats = feats.flatten(2)                    # B x C x N_cvae
        feats = feats.transpose(1, 2).contiguous()  # B x N_cvae x C
        feats = self.cvae_proj(feats)               # B x N_cvae x 1024
        return feats

    def forward(self, input_modal, text_embeddings=None):
        """
        Compute CVAE encoder outputs (mu, logvar) and decoder reconstruction.
        
        Args:
            input_modal: Image tensor with shape (B, channels, H, W)
            text_embeddings: Optional text embeddings with shape (B, text_embedding_dim)
        """
        # Initialize CVAE outputs as None
        mu, logvar, reconstructed = None, None, None
        z = None

        tensor_first = input_modal[:, :9, :, :]
        offset = 9

        tensor_ae = None
        tensor_anysat = None
        tensor_terramind = None

        if self.alphaearth_flag:
            tensor_ae = input_modal[:, offset:offset+64, :50, :50]   # B x 64 x 50 x 50
            offset += 64
        if self.anysat_flag:
            tensor_anysat = input_modal[:, offset:offset+1536, :50, :50]  # B x 1536 x 50 x 50
            offset += 1536
        if self.terramind_flag:
            tensor_terramind = input_modal[:, offset:offset+384, :14, :14]  # B x 384 x 14 x 14
        
        # Only run CVAE if enabled
        if self.cvae:
            if self.cvae_treatment_source == "alphaearth":
                treatment = tensor_ae
            else:
                treatment = tensor_first

            if self.cvae_covariate_source == "alphaearth":
                covariate = tensor_ae
            else:
                covariate = tensor_first
                        
            # Encoder: compute posterior q(Z|A,X,T) where T is text embeddings
            mu, logvar = self.cvae_encoder(treatment, covariate, text_embeddings=text_embeddings)  # (B, latent_d, latent_h, latent_w)
            
            # Reparameterize: Z ~ q(Z|A,X,T)
            z = reparameterize(mu, logvar)  # (B, latent_d, latent_h, latent_w)
            
            # Decoder: reconstruct A|X,Z,T with VMF likelihood
            # Normalize output if alphaearth is the treatment source (unit norm per pixel)
            normalize_output = (self.cvae_treatment_source == "alphaearth")
            reconstructed_mean, reconstructed_kappa = self.cvae_decoder(z, covariate, text_embeddings=text_embeddings, normalize_output=normalize_output)  # (B, treatment_channels, H, W), (B, 1, H, W)
        
        # Process main images
        tensor_rs, tensor_dsm, tensor_giu = torch.chunk(tensor_first, 3, dim=1)

        features_rs = self.model(tensor_rs)
        features_dsm = self.model(tensor_dsm)
        features_giu = self.model(tensor_giu)

        out_rs = self._dense_head(features_rs)   # B x N x 1024
        out_dsm = self._dense_head(features_dsm)  # B x N x 1024
        out_giu = self._dense_head(features_giu)  # B x N x 1024

        # Process alphaearth / anysat / terramind if present
        out_ae = self._ae_head(tensor_ae) if hasattr(self, 'ae_conv') and tensor_ae is not None else None
        out_anysat = self._anysat_head(tensor_anysat) if hasattr(self, 'anysat_conv') and tensor_anysat is not None else None
        out_terramind = self._terramind_head(tensor_terramind) if hasattr(self, 'terramind_conv') and tensor_terramind is not None else None

        # Process CVAE if enabled
        out_latent = None
        if self.cvae and z is not None:
            out_latent = self._cvae_head(mu.detach())

        # Concatenate outputs
        parts = [out_rs, out_dsm, out_giu]
        for extra in (out_ae, out_anysat, out_terramind, out_latent):
            if extra is not None:
                parts.append(extra)
        out = torch.cat(parts, dim=1)
        
        if self.cvae:
            self.latent = mu.detach().cpu()
            return mu, logvar, reconstructed_mean, reconstructed_kappa, out
        else:
            return out
