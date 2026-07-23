"""
The classes and functions in this module are adapted from Huggingface implementation: utils_mmimdb.py, which can be
found here: https://github.com/huggingface/transformers/blob/8ea412a86faa8e9edeeb6b5c46b08def06aa03ea/examples/research_projects/mm-imdb/utils_mmimdb.py

The ImageEncoderDenseNet class is modified from the original ImageEncoder class to be based on pre-trained DenseNet
instead of ResNet and to be albe to load saved pre-trained weights.

The forward function is also modified according to the forward function of the DenseNet model liste here:

Original forward function of DenseNet

def forward(self, x):
    features = self.features(x)
    out = F.relu(features, inplace=True)
    out = F.adaptive_avg_pool2d(out, (1, 1))
    out = torch.flatten(out, 1)
    out = self.classifier(out)
    return out

"""
import json
import os
from collections import Counter
import logging
import cv2
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image 
from torch.utils.data import Dataset
from torchvision.transforms import transforms


logger = logging.getLogger(__name__)

# directories and data filenames
MMBT_DIR_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Data root: defaults to <repo>/data_livability, override with $LIVABILITY_DATA_DIR
DATA_DIR = os.environ.get("LIVABILITY_DATA_DIR", os.path.join(MMBT_DIR_PARENT, "data_livability"))
#JSONL_DATA_DIR = os.path.join(DATA_DIR, "json")
JSONL_DATA_DIR = os.path.join(DATA_DIR, "json")
#IMG_DATA_DIR = os.path.join(DATA_DIR, "NLMCXR_png_frontal")
IMG_DATA_DIR = os.path.join(DATA_DIR, "Liva_RS") # img_AMS_9_instances  Utrecht_rs "img"
DSM_DATA_DIR = os.path.join(DATA_DIR, "Liva_DSM") # dsm_AMS_9_instances  Utrecht_dsm "dsm"
GIU_DATA_DIR = os.path.join(DATA_DIR, "Liva_GIU_RGB") # 
ALPHAEARTH_DATA_DIR = os.path.join(DATA_DIR, "alphaearth")  # only used if alphaearth = True
ANYSAT_DATA_DIR = os.path.join(DATA_DIR, "anysat")          # only used if anysat = True
TERRAMIND_DATA_DIR = os.path.join(DATA_DIR, "terramind")    # only used if terramind = True

class JsonlDataset(Dataset):
    def __init__(self, jsonl_data_path, img_dir, dsm_dir, giu_dir, alphaearth_dir, anysat_dir, terramind_dir, tokenizer, transforms, transforms_gray, transforms_giu, transforms_alphaearth, transforms_anysat, transforms_terramind, transforms_raw, transforms_gray_raw, transforms_giu_raw, labels, max_seq_length, alphaearth, anysat, terramind, alphazero, terrazero=False, anyzero=False):
        self.data = [json.loads(line) for line in open(jsonl_data_path)]
        # self.data_dir = os.path.dirname(data_path)
        self.img_data_dir = img_dir
        self.dsm_data_dir = dsm_dir
        self.giu_data_dir = giu_dir
        self.alphaearth_data_dir = alphaearth_dir
        self.anysat_data_dir = anysat_dir
        self.terramind_data_dir = terramind_dir

        self.tokenizer = tokenizer
        self.labels = labels
        self.n_classes = 6
        self.max_seq_length = max_seq_length

        # for image normalization for DenseNet
        self.transforms = transforms
        self.transforms_gray = transforms_gray
        self.transforms_giu = transforms_giu
        self.transforms_alphaearth = transforms_alphaearth
        self.transforms_anysat = transforms_anysat
        self.transforms_terramind = transforms_terramind

        self.transforms_raw = transforms_raw
        self.transforms_gray_raw = transforms_gray_raw
        self.transforms_giu_raw = transforms_giu_raw

        self.alphaearth = alphaearth
        self.anysat = anysat
        self.terramind = terramind
        self.alphazero = alphazero
        self.terrazero = terrazero
        self.anyzero = anyzero

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        sentence = torch.LongTensor(self.tokenizer.encode(self.data[index]["text"], add_special_tokens=True))
        start_token, sentence, end_token = sentence[0], sentence[1:-1], sentence[-1]
        sentence = sentence[:self.max_seq_length]

        if self.n_classes > 2:
            # multiclass

            label = torch.zeros(self.n_classes)  
            label[0] = float(self.data[index]["lbm"])          
            label[1] = float(self.data[index]["fys"])
            label[2] = float(self.data[index]["onv"])
            label[3] = float(self.data[index]["soc"])
            label[4] = float(self.data[index]["vrz"])
            label[5] = float(self.data[index]["won"])
                                 
        #    label[0] = torch.LongTensor([self.labels.index(self.data[index]["lbm"])])           
        #    label[1] = torch.LongTensor([self.labels.index(self.data[index]["fys"])])
        #    label[2] = torch.LongTensor([self.labels.index(self.data[index]["onv"])])
        #    label[3] = torch.LongTensor([self.labels.index(self.data[index]["soc"])])
        #    label[4] = torch.LongTensor([self.labels.index(self.data[index]["vrz"])])
        #    label[5] = torch.LongTensor([self.labels.index(self.data[index]["won"])])
         #   label[6] = torch.LongTensor([self.labels.index(self.data[index]["afw"])])

            
            
     
            #label[self.labels.index(self.data[index]["label"])] = 1
        else:
            label = torch.LongTensor([self.labels.index(self.data[index]["label"])])


        rs = Image.open(os.path.join(self.img_data_dir, self.data[index]["img"])).convert("RGB")
        rs_transform = self.transforms(rs)
        rs_raw = self.transforms_raw(rs)


        dsm = cv2.imread(os.path.join(self.dsm_data_dir, self.data[index]["dsm"]),cv2.IMREAD_GRAYSCALE) # cv2.IMREAD_GRAYSCALE: grayscale image, cv2.COLOR_GRAY2RGB: convert to RGB
        dsm = cv2.cvtColor(dsm, cv2.COLOR_GRAY2RGB)

        dsm = Image.fromarray(dsm)
        dsm_transform = self.transforms_gray(dsm)
        dsm_raw = self.transforms_gray_raw(dsm)
        # result1 = torch.cat((rs,dsm),dim=0)

        giu = Image.open(os.path.join(self.giu_data_dir, self.data[index]["giu"])).convert("RGB")
        giu_transform = self.transforms_giu(giu)
        giu_raw = self.transforms_giu_raw(giu)

        images = torch.cat((rs_transform,dsm_transform,giu_transform),dim=0)
        images_raw = torch.cat((rs_raw,dsm_raw,giu_raw),dim=0)

        cur_fname_alpha = [
            (self.data[index]["img"], "RS"),
            (self.data[index]["giu"], "GIU"),
            (self.data[index]["dsm"], "DSM")
        ]
        cur_fname_valid = [s for s in cur_fname_alpha if s[0] != "NULL.tif"]
        if not cur_fname_valid:
            ref = self.data[index].get("ref_img", None)
            if ref is None:
                raise ValueError(f"All images are NULL.tif for record {index} but no ref_img field found.")
            cur_fname_valid = [(ref, "RS")]

        if self.alphaearth:
            alphaearth_name = os.path.join(self.alphaearth_data_dir, cur_fname_valid[0][0].replace(cur_fname_valid[0][1], "alphaearth").replace(".tif", "_2020.npz"))
            alphaearth = np.load(alphaearth_name)['image_data']
            if self.alphazero:
                alphaearth = np.zeros_like(alphaearth)
            alphaearth = self.transforms_alphaearth(alphaearth)
            images = torch.cat((images, alphaearth), dim=0)

        if self.anysat:
            anysat_name = os.path.join(self.anysat_data_dir, cur_fname_valid[0][0].replace(cur_fname_valid[0][1], "anysat").replace(".tif", "_2020.npz"))
            anysat_emb = np.load(anysat_name)['image_data']  # (1536, 24, 24)
            if self.anyzero:
                anysat_emb = np.zeros_like(anysat_emb)
            anysat_emb = self.transforms_anysat(anysat_emb)
            images = torch.cat((images, anysat_emb), dim=0)

        if self.terramind:
            terramind_name = os.path.join(self.terramind_data_dir, cur_fname_valid[0][0].replace(cur_fname_valid[0][1], "terramind").replace(".tif", "_2020.npz"))
            terramind_emb = np.load(terramind_name)['image_data']  # (384, 14, 14)
            if self.terrazero:
                terramind_emb = np.zeros_like(terramind_emb)
            terramind_emb = self.transforms_terramind(terramind_emb)
            images = torch.cat((images, terramind_emb), dim=0)

        return {
            "image_start_token": start_token,
            "image_end_token": end_token,
            "sentence": sentence,
            "image": images,
            "label": label,
            "images_raw": images_raw
        }


    def get_label_frequencies(self): ##
        label_freqs = Counter()
        for row in self.data:
            label_freqs.update([row["label"]])
        return label_freqs


def collate_fn(batch):
    """
    Specify batching for the torch Dataloader function

    :param batch: each batch of the JsonlDataset
    :return: text tensor, attention mask tensor, img tensor, modal start token, modal end token, label
    """
    lens = [len(row["sentence"]) for row in batch]
    bsz, max_seq_len = len(batch), max(lens)

    mask_tensor = torch.zeros(bsz, max_seq_len, dtype=torch.long)
    text_tensor = torch.zeros(bsz, max_seq_len, dtype=torch.long)

    for i_batch, (input_row, length) in enumerate(zip(batch, lens)):
        text_tensor[i_batch, :length] = input_row["sentence"]
        mask_tensor[i_batch, :length] = 1

    img_tensor = torch.stack([row["image"] for row in batch])
    tgt_tensor = torch.stack([row["label"] for row in batch])
    img_start_token = torch.stack([row["image_start_token"] for row in batch])
    img_end_token = torch.stack([row["image_end_token"] for row in batch])
    img_raw_tensor = torch.stack([row["images_raw"] for row in batch])
    ###########
    #dsm_tensor = torch.stack([row["dsm"] for row in batch])


    return text_tensor, mask_tensor, img_tensor, img_start_token, img_end_token, tgt_tensor, img_raw_tensor #


def collate_fn_mask_all_text(batch):
    """
    Specify batching for the torch Dataloader function

    :param batch: each batch of the JsonlDataset
    :return: text tensor, attention mask tensor, img tensor, modal start token, modal end token, label
    """
    lens = [len(row["sentence"]) for row in batch]
    bsz, max_seq_len = len(batch), max(lens)

    mask_tensor = torch.zeros(bsz, max_seq_len, dtype=torch.long)
    text_tensor = torch.zeros(bsz, max_seq_len, dtype=torch.long)

    for i_batch, (input_row, length) in enumerate(zip(batch, lens)):
        text_tensor[i_batch, :length] = input_row["sentence"]
        #mask_tensor[i_batch, :length] = 1

    img_tensor = torch.stack([row["image"] for row in batch])    
    tgt_tensor = torch.stack([row["label"] for row in batch])
    img_start_token = torch.stack([row["image_start_token"] for row in batch])
    img_end_token = torch.stack([row["image_end_token"] for row in batch])
    img_raw_tensor = torch.stack([row["images_raw"] for row in batch])

    return text_tensor, mask_tensor, img_tensor, img_start_token, img_end_token, tgt_tensor, img_raw_tensor


def get_multiclass_labels(): 
    """
    0: lbm
    1: fys
    2: onv
    3: soc 
    4: vrz
    5: won 
 #   6: 



    # Because there are 14 classes


    :return:
    """

    
    return [ 0, 1, 2, 3, 4, 5 ]



def get_labels():
    """
    0: R1
    1: R2
    2: R3
    3: R4
    4: B0
    5: B1
    6: B2
    7: B3
    8: B5
    9: B6
    10: I
    11: A1
    12: A3
    13: A4

    



    :return:
    """

    
    return [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13 ]


def get_image_transforms():
    """
    Transforms image tensor, resize, center, and normalize according to the Mean and Std specific to the DenseNet model
    :return: None
    """
    return transforms.Compose(
        [
          #  transforms.Resize(250),
            transforms.CenterCrop(250),
            transforms.Resize((224,224)), # Resize cropped image to 224x224 to match DSM dimensions; concatenate them. Default padding is 0.
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]
    )

def get_image_transforms_raw(centercrop=250):
    """
    Transforms image tensor, resize, center, and normalize according to the Mean and Std specific to the DenseNet model
    :return: None
    """
    return transforms.Compose(
        [
          #  transforms.Resize(250),
            transforms.CenterCrop(centercrop),
            transforms.Resize((224,224)), # Resize cropped image to 224x224 to match DSM dimensions; concatenate them. Default padding is 0.
            transforms.ToTensor()
        ]
    )    
def get_image_transforms_gray():
    """
    Transforms image tensor, resize, center, and normalize according to the Mean and Std specific to the DenseNet model
    :return: None
    """
    return transforms.Compose(
        [
          #  transforms.Resize(250),
            transforms.CenterCrop(200),
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]
    )

def get_image_transforms_giu():
    """
    Transforms image tensor, resize, center, and normalize according to the Mean and Std specific to the DenseNet model
    :return: None
    """
    return transforms.Compose(
        [
         #   transforms.Resize(50),
            transforms.CenterCrop(50),
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]
    )

def get_image_transforms_alphaearth():
    """
    Transforms alphaearth numpy array (floating point) to tensor and pads to 224x224.
    Note: Alphaearth data is loaded as numpy array from .npz file with float values
    :return: Custom transform function (not transforms.Compose since we need tensor operations)
    """
    def transform(img_array):
        # Convert numpy array to tensor (preserves float values)
        if isinstance(img_array, np.ndarray):
            img_tensor = torch.from_numpy(img_array).float()
        else:
            img_tensor = img_array.float()
                
        # Center crop to 50x50
        _, h, w = img_tensor.shape
        crop_size = 50
        start_h = (h - crop_size) // 2
        start_w = (w - crop_size) // 2
        img_tensor = img_tensor[:, start_h:start_h+crop_size, start_w:start_w+crop_size]
        
        # Pad (left, right, top, bottom) to 224x224
        # From 50x50 to 224x224: pad (0, 174, 0, 174)
        img_tensor = F.pad(img_tensor, (0, 174, 0, 174), mode='constant', value=0)
        
        return img_tensor
    
    return transform
def get_image_transforms_anysat():
    """
    Transforms anysat numpy array (1536, 51, 51): center crop to 50x50, pad to (1536, 224, 224).
    Mirrors the AlphaEarth transform exactly.
    """
    def transform(img_array):
        if isinstance(img_array, np.ndarray):
            img_tensor = torch.from_numpy(img_array).float()
        else:
            img_tensor = img_array.float()
        # Center crop to 50x50
        _, h, w = img_tensor.shape
        crop_size = 50
        start_h = (h - crop_size) // 2
        start_w = (w - crop_size) // 2
        img_tensor = img_tensor[:, start_h:start_h+crop_size, start_w:start_w+crop_size]
        # Pad to 224x224
        img_tensor = F.pad(img_tensor, (0, 174, 0, 174), mode='constant', value=0)
        return img_tensor
    return transform


def get_image_transforms_terramind():
    """
    Transforms terramind numpy array (384, 14, 14): pad to (384, 224, 224).
    No crop needed since 14x14 is smaller than 50x50.
    """
    def transform(img_array):
        if isinstance(img_array, np.ndarray):
            img_tensor = torch.from_numpy(img_array).float()
        else:
            img_tensor = img_array.float()
        # Pad from 14x14 to 224x224
        img_tensor = F.pad(img_tensor, (0, 210, 0, 210), mode='constant', value=0)
        return img_tensor
    return transform


def load_examples(tokenizer, wandb_config, alphaearth, alphazero, anysat=False, terramind=False, anyzero=False, terrazero=False, evaluate=False, test=False, data_dir=JSONL_DATA_DIR, img_dir=IMG_DATA_DIR, dsm_dir=DSM_DATA_DIR, giu_dir=GIU_DATA_DIR, alphaearth_dir=ALPHAEARTH_DATA_DIR, anysat_dir=ANYSAT_DATA_DIR, terramind_dir=TERRAMIND_DATA_DIR):
    """

    :param tokenizer: BERT tokenizer of choice
    :param wandb_config: wandb.config, which needs to contain file names of validation, test, and train files
    :param evaluate: True if loading Dataset for evaluating on validation or test set, False for Training
    :param test: True ONLY if loading Test Dataset, False if evaluating on validation set; if evaluate = False, test has to be False
    :param data_dir: Path to jsonl data directory e.g. "data/json"
    :param img_dir: Path to image directory e.g. "NLMCXR_png_frontal"
    :return: JasonlDataset derived from Torch Dataset class
    """
    if evaluate and not test:
        path = os.path.join(data_dir, wandb_config.val_file)
    elif evaluate and test:
        path = os.path.join(data_dir, wandb_config.test_file)
    elif not evaluate and not test:
        path = os.path.join(data_dir, wandb_config.train_file)
    else:
        # shouldn't get here not evaluate and test?
        raise ValueError("invalid data file option!!")

    img_transforms = get_image_transforms()
    img_transforms_gray = get_image_transforms_gray()
    img_transforms_giu = get_image_transforms_giu()
    img_transforms_alphaearth = get_image_transforms_alphaearth()
    img_transforms_anysat = get_image_transforms_anysat()
    img_transforms_terramind = get_image_transforms_terramind()

    img_transforms_raw = get_image_transforms_raw(250)
    img_transforms_gray_raw = get_image_transforms_raw(200)
    img_transforms_giu_raw = get_image_transforms_raw(50)

    if wandb_config.multiclass:
        labels = get_multiclass_labels()
    else:
        labels = get_labels()

    dataset = JsonlDataset(
        path, img_dir, dsm_dir, giu_dir, alphaearth_dir, anysat_dir, terramind_dir,
        tokenizer,
        img_transforms, img_transforms_gray, img_transforms_giu,
        img_transforms_alphaearth, img_transforms_anysat, img_transforms_terramind,
        img_transforms_raw, img_transforms_gray_raw, img_transforms_giu_raw,
        labels, wandb_config.max_seq_length - wandb_config.num_image_embeds - 2,
        alphaearth, anysat, terramind, alphazero, terrazero, anyzero,
    )

    logger.info(f"JsonlDataset from {path}\n")

    return dataset


def get_multiclass_criterion(jsonl_dataset_obj):
    label_freqs = jsonl_dataset_obj.get_label_frequencies()
    freqs = [label_freqs[label] for label in jsonl_dataset_obj.labels]
    label_weights = (torch.tensor(freqs, dtype=torch.float) / len(jsonl_dataset_obj)) ** -1
    return nn.BCEWithLogitsLoss(pos_weight=label_weights.cuda())
