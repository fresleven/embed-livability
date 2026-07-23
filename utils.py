import os
import json
import logging

import numpy as np
from scipy.stats import linregress, gaussian_kde, kendalltau
import matplotlib.pyplot as plt
import random
import glob

from sklearn.metrics import f1_score

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm, trange

from transformers import WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup

# try:
#     from torch.utils.tensorboard import SummaryWriter
# except ImportError:
#     from tensorboardX import SummaryWriter

from textBert_utils import set_seed
from MMBT_liva.mmbt_utils_liva_0318 import load_examples, collate_fn, get_multiclass_criterion
from MMBT_liva.image_liva import CVAEEncoder, CVAEDecoder
from cvae_utils import CVAELoss, PosteriorPredictiveCheck, kl_warmup_schedule, reparameterize, get_treatment_and_covariates

# Module-level logger for this module; training script configures handlers
logger = logging.getLogger(__name__)


# Assume `tgts` and `preds` are already defined and computed
# `tgts` and `preds` are multi-dimensional tensors where each column corresponds to a different label
def plot_density_scatter(tgts, preds, label, ax, xlim, ylim):
    """Plot a density scatter and set axis ranges."""
    # compute density
    xy = np.vstack([tgts, preds])
    z = gaussian_kde(xy)(xy)
    
    # plot scatter
    scatter = ax.scatter(tgts, preds, c=z, s=5, edgecolor=None, cmap='viridis')
    
    # add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Density')
    
    ax.set_xlabel(f'Ground Truth of {label}')
    ax.set_ylabel(f'Predicted value of {label}')
    ax.set_title(f'Density Scatter Plot of {label}')
    ax.grid(True)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect('equal', adjustable='box')

def calculate_tau_with_p_value(y_true, y_pred):
    """Compute Kendall's tau and its p-value."""
    tau, p_value = kendalltau(y_true, y_pred)
    return tau, p_value


def train(args, train_dataset, model, tokenizer, alphaearth, anysat=False, terramind=False, resume_from_step=0, resume_checkpoint_dir=None):
    """ Train the model with optional C-VAE for confounder reconstruction """
    
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.train_batch_size,
        collate_fn=collate_fn,
    )

    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Initialize C-VAE loss function if enabled
    cvae_loss_fn = None
    total_warmup_steps = 0
    if args.use_cvae:
        # cvae_latent_h = cvae_latent_w = 50 if args.cvae_treatment_source == "alphaearth" else 224
        cvae_latent_h = cvae_latent_w = args.cvae_latent_hw
        cvae_loss_fn = CVAELoss(beta=0, prior_type='laplacian_gmrf', latent_h = cvae_latent_h, latent_w = cvae_latent_w, treatment = args.cvae_treatment_source)
        # Calculate total warmup steps (across all epochs)
        total_warmup_steps = int(len(train_dataloader) / args.gradient_accumulation_steps * args.cvae_kl_warmup_epochs)
        logger.info("C-VAE initialized with latent shape: (%d, %d, %d)", 
                   cvae_latent_h, cvae_latent_w, args.cvae_latent_d)
        logger.info("Treatment source: %s, Covariate source: %s", 
                   args.cvae_treatment_source, args.cvae_covariate_source)
        logger.info("KL warmup schedule: %d total steps over %d epochs", total_warmup_steps, args.cvae_kl_warmup_epochs)

    # Prepare optimizers for CVAE and MMBT models (separate optimizers)
    no_decay = ["bias", "LayerNorm.weight"]
    
    # CVAE parameters
    cvae_params = []
    mmbt_params = []
    
    for n, p in model.named_parameters():
        if "cvae" in n.lower():
            cvae_params.append((n, p))
        else:
            mmbt_params.append((n, p))
    
    # CVAE optimizer
    if args.use_cvae and cvae_params:
        cvae_optimizer_grouped_parameters = [
            {
                "params": [p for n, p in cvae_params if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {"params": [p for n, p in cvae_params if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]
        cvae_optimizer = AdamW(cvae_optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        cvae_scheduler = get_linear_schedule_with_warmup(
            cvae_optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
        )
    else:
        cvae_optimizer = None
        cvae_scheduler = None
    
    # MMBT optimizer
    mmbt_optimizer_grouped_parameters = [
        {
            "params": [p for n, p in mmbt_params if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in mmbt_params if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    mmbt_optimizer = AdamW(mmbt_optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    mmbt_scheduler = get_linear_schedule_with_warmup(
        mmbt_optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # Restore optimizer and scheduler states when resuming
    if resume_checkpoint_dir is not None:
        opt_path = os.path.join(resume_checkpoint_dir, "optimizer.pt")
        sch_path = os.path.join(resume_checkpoint_dir, "scheduler.pt")
        if os.path.exists(opt_path):
            mmbt_optimizer.load_state_dict(torch.load(opt_path, map_location=args.device))
            mmbt_scheduler.load_state_dict(torch.load(sch_path, map_location=args.device))
            logger.info("Loaded optimizer/scheduler state from %s", resume_checkpoint_dir)
        if cvae_optimizer is not None:
            cvae_opt_path = os.path.join(resume_checkpoint_dir, "cvae_optimizer.pt")
            cvae_sch_path = os.path.join(resume_checkpoint_dir, "cvae_scheduler.pt")
            if os.path.exists(cvae_opt_path):
                cvae_optimizer.load_state_dict(torch.load(cvae_opt_path, map_location=args.device))
                cvae_scheduler.load_state_dict(torch.load(cvae_sch_path, map_location=args.device))
                logger.info("Loaded CVAE optimizer/scheduler state from %s", resume_checkpoint_dir)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Total train batch size = %d", args.train_batch_size * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Use C-VAE: %s", args.use_cvae)

    global_step = resume_from_step
    tr_loss = 0.0
    tr_cvae_loss = 0.0
    tr_recon_cvae_loss = 0.0
    tr_kl_cvae_loss = 0.0
    best_eval_metric = float('inf') if args.multiclass else -float('inf')
    n_no_improve = 0

    # CVAE convergence tracking
    best_cvae_loss = float('inf')
    cvae_no_improve_count = 0
    cvae_frozen = False
    cvae_patience = 5  # Number of logging steps without improvement before freezing

    # Compute how many epochs/steps are already done when resuming
    steps_per_epoch = len(train_dataloader) // args.gradient_accumulation_steps
    resume_epoch = resume_from_step // steps_per_epoch if steps_per_epoch > 0 else 0
    resume_step  = resume_from_step % steps_per_epoch if steps_per_epoch > 0 else 0

    model.train()
    model.zero_grad()
    mmbt_optimizer.zero_grad()
    if cvae_optimizer is not None:
        cvae_optimizer.zero_grad()

    train_iterator = trange(int(args.num_train_epochs), desc="Epoch")
    set_seed(args)

    for epoch_idx in train_iterator:
        if epoch_idx < resume_epoch:
            continue  # skip completed epochs

        epoch_iterator = tqdm(train_dataloader, desc="Training Batch Iteration", disable=True, dynamic_ncols=False, position=0, leave=True)

        for step, batch in enumerate(epoch_iterator):
            # Skip steps already completed in the first resumed epoch
            if epoch_idx == resume_epoch and step < resume_step:
                continue
            batch = tuple(t.to(args.device) for t in batch)
            
            labels = batch[5]
            input_ids = batch[0]
            input_modal = batch[2]
            attention_mask = batch[1]
            modal_start_tokens = batch[3]
            modal_end_tokens = batch[4]
            input_modal_raw = batch[6]
                                        
            if args.multiclass:
                outputs = model(
                    input_modal,
                    input_ids=input_ids,
                    modal_start_tokens=modal_start_tokens,
                    modal_end_tokens=modal_end_tokens,
                    attention_mask=attention_mask,
                    token_type_ids=None,
                    modal_token_type_ids=None,
                    position_ids=None,
                    modal_position_ids=None,
                    head_mask=None,
                    inputs_embeds=None,
                    labels=None,
                    return_dict=True
                )
            else:
                outputs = model(
                    input_modal_with_latent,
                    input_ids=input_ids,
                    modal_start_tokens=modal_start_tokens,
                    modal_end_tokens=modal_end_tokens,
                    attention_mask=attention_mask,
                    token_type_ids=None,
                    modal_token_type_ids=None,
                    position_ids=None,
                    modal_position_ids=None,
                    head_mask=None,
                    inputs_embeds=None,
                    labels=None,
                    return_dict=True
                )

            if args.use_cvae:
                outputs, mu, logvar, reconstructed_mean, reconstructed_kappa = outputs
                treatments, _ = get_treatment_and_covariates(batch, args, 6)
                # Compute C-VAE loss
                loss_cvae, recon_loss, kl_loss = cvae_loss_fn(reconstructed_mean, reconstructed_kappa, treatments, mu, logvar)
                
                tr_cvae_loss += loss_cvae.item()
                tr_recon_cvae_loss += recon_loss.item()
                tr_kl_cvae_loss += kl_loss.item()
            
            logits = outputs.logits
            
            # Compute MMBT loss
            if args.multiclass:
                my_loss = torch.nn.L1Loss()
                loss_mmbt = my_loss(logits, labels)
            else:
                criterion = get_multiclass_criterion(train_dataset)
                loss_mmbt = criterion(logits, labels)
            
            # ===== Combined Loss and Single Backward Pass =====
            # Combine losses: L_total = L_Y + lambda * L_A
            if args.use_cvae:
                loss = loss_mmbt + loss_cvae
            else:
                loss = loss_mmbt
            
            if args.n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            tr_loss += loss_mmbt.item()
            
            if (step + 1) % args.gradient_accumulation_steps == 0:
                # Update KL weight with per-step granularity during warmup
                if cvae_loss_fn is not None and global_step < total_warmup_steps:
                    beta = min(args.cvae_kl_weight, global_step / total_warmup_steps * args.cvae_kl_weight)
                    cvae_loss_fn.set_beta(beta)
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                # Step both optimizers (skip CVAE if frozen)
                mmbt_optimizer.step()
                mmbt_scheduler.step()
                if cvae_optimizer is not None and not cvae_frozen:
                    cvae_optimizer.step()
                    cvae_scheduler.step()
                
                model.zero_grad()
                
                global_step += 1

                if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logs = {}
                    if args.evaluate_during_training:
                        results = evaluate(args, model, tokenizer, alphaearth, alphazero=False, anysat=anysat, terramind=terramind)
                        for key, value in results.items():
                            eval_key = "eval_{}".format(key)
                            logs[eval_key] = value

                    loss_scalar = tr_loss / args.logging_steps
                    learning_rate_scalar = mmbt_scheduler.get_last_lr()[0]
                    logs["learning_rate"] = learning_rate_scalar
                    logs["training_loss"] = loss_scalar
                    
                    if cvae_loss_fn is not None:
                        avg_cvae_loss = tr_cvae_loss / args.logging_steps
                        logs["cvae_loss"] = avg_cvae_loss
                        logs["cvae_recon_loss"] = tr_recon_cvae_loss / args.logging_steps
                        logs["cvae_kl_loss"] = tr_kl_cvae_loss / args.logging_steps
                        logs["cvae_frozen"] = cvae_frozen
                        
                        # Check CVAE convergence and freeze if converged
                        if not cvae_frozen and args.use_cvae and args.cvae_kl_weight == beta:
                            if avg_cvae_loss < best_cvae_loss:
                                best_cvae_loss = avg_cvae_loss
                                cvae_no_improve_count = 0
                            else:
                                cvae_no_improve_count += 1
                            
                            # Freeze CVAE parameters if no improvement for cvae_patience logging steps
                            if cvae_no_improve_count >= cvae_patience:
                                cvae_frozen = True
                                # Freeze CVAE parameters
                                for n, p in model.named_parameters():
                                    if "cvae" in n.lower():
                                        p.requires_grad = False
                                logger.info("CVAE converged at global_step %d. Freezing CVAE parameters to prevent overfitting.", global_step)
                                logger.info("Best CVAE loss: %.6f", best_cvae_loss)
                        
                        tr_cvae_loss = 0.0
                        tr_recon_cvae_loss = 0.0
                        tr_kl_cvae_loss = 0.0

                    tr_loss = 0.0

                    logs_serializable = {key: float(value.cpu().numpy()) if isinstance(value, torch.Tensor) else value for key, value in logs.items()}

                    cur_val_file = os.path.join(args.output_dir, "checkpoint-{}".format(global_step), "eval_results_val.txt")
                    os.makedirs(os.path.dirname(cur_val_file), exist_ok=True)
                    with open(cur_val_file, "w") as f:
                        json.dump(logs_serializable, f, indent=2)
                    
                    print(json.dumps(logs_serializable))

                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    print ("global_step",global_step)
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir, exist_ok=True)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    torch.save(model_to_save.state_dict(), os.path.join(output_dir, WEIGHTS_NAME))
                    # Save optimizer and scheduler state for full resume
                    torch.save(mmbt_optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(mmbt_scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    if cvae_optimizer is not None:
                        torch.save(cvae_optimizer.state_dict(), os.path.join(output_dir, "cvae_optimizer.pt"))
                        torch.save(cvae_scheduler.state_dict(), os.path.join(output_dir, "cvae_scheduler.pt"))
                    logger.info("Saving model checkpoint to %s", output_dir)

        results = evaluate(args, model, tokenizer, alphaearth, alphazero=False, anysat=anysat, terramind=terramind)
        if args.multiclass:
            eval_result = results["rmse_lbm"]
        else:
            eval_result = results["macro_f1"]

        is_better = (eval_result < best_eval_metric) if args.multiclass else (eval_result > best_eval_metric)

        if is_better:
            best_eval_metric = eval_result
            n_no_improve = 0
        else:
            n_no_improve += 1

        if n_no_improve > args.patience:
            train_iterator.close()
            break

    logs = {}
    if args.evaluate_during_training:
        results = evaluate(args, model, tokenizer, alphaearth, alphazero=False, anysat=anysat, terramind=terramind)
        for key, value in results.items():
            eval_key = "eval_{}".format(key)
            logs[eval_key] = value

    loss_scalar = tr_loss / max(args.logging_steps, 1)
    learning_rate_scalar = mmbt_scheduler.get_last_lr()[0]
    logs["learning_rate"] = learning_rate_scalar
    logs["training_loss"] = loss_scalar

    if cvae_loss_fn is not None:
        logs["cvae_loss"] = tr_cvae_loss / args.logging_steps
        logs["cvae_recon_loss"] = tr_recon_cvae_loss / args.logging_steps
        logs["cvae_kl_loss"] = tr_kl_cvae_loss / args.logging_steps

    logs_serializable = {key: float(value.cpu().numpy()) if isinstance(value, torch.Tensor) else value 
                        for key, value in logs.items()}
    cur_val_file = os.path.join(args.output_dir, "checkpoint-final", "eval_results_val.txt")
    os.makedirs(os.path.dirname(cur_val_file), exist_ok=True)
    with open(cur_val_file, "w") as f:
        json.dump(logs_serializable, f, indent=2)

    return global_step, tr_loss / max(global_step, 1)

def evaluate(args, model, tokenizer, alphaearth, alphazero, anysat=False, terramind=False, anyzero=False, terrazero=False, evaluate=True, test=False, prefix=""):
    """
    Evaluate the model with optional C-VAE confounder reconstruction and posterior predictive checks.
    
    Args:
        args: arguments
        model: MMBT model
        tokenizer: text tokenizer
        alphaearth: whether to use alphaearth data
        evaluate: whether this is eval (vs test)
        test: whether this is test set
        prefix: prefix for output files
    """
    
    # Initialize C-VAE loss function if enabled (same as in train)
    cvae_loss_fn = None
    if args.use_cvae:
        # cvae_latent_h = cvae_latent_w = 50 if args.cvae_treatment_source == "alphaearth" else 224
        cvae_latent_h = cvae_latent_w = args.cvae_latent_hw
        cvae_loss_fn = CVAELoss(beta=0, prior_type='laplacian_gmrf', latent_h=cvae_latent_h, latent_w=cvae_latent_w, treatment = args.cvae_treatment_source)
    
    eval_output_dir = args.output_dir
    eval_dataset = load_examples(tokenizer, args, alphaearth=alphaearth, alphazero=alphazero, anysat=anysat, terramind=terramind, anyzero=anyzero, terrazero=terrazero, evaluate=evaluate, test=test)

    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=collate_fn
    )

    # Eval!
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    eval_cvae_loss = 0.0
    eval_recon_cvae_loss = 0.0
    eval_kl_cvae_loss = 0.0
    nb_eval_steps = 0
    preds = []
    out_label_ids = []
    all_treatments = []
    all_covariates = []
    latents = []
    attentions = []
    
    # For posterior predictive checks
    ppc_results = None
    ppc_all_p_values = []
    ppc_all_mean_p_values = []
    ppc_all_is_valid_samples = []
    ppc_all_test_stats_true = []
    ppc_batch_count = 0
    if args.use_cvae:
        ppc_checker = PosteriorPredictiveCheck(discrepancy_type = 'vmf_loglikelihood' if args.cvae_treatment_source == "alphaearth" else "marginal_loglikelihood")
    
    for batch in tqdm(eval_dataloader, desc="Evaluating", disable=True, dynamic_ncols=False, position=0, leave=True):
        model.eval()

        with torch.no_grad():
            batch = tuple(t.to(args.device) for t in batch)
            labels = batch[5]
            input_ids = batch[0]
            input_modal = batch[2]
            attention_mask = batch[1]
            modal_start_tokens = batch[3]
            modal_end_tokens = batch[4]
            input_modal_raw = batch[6]
            
            if args.multiclass:
                outputs = model(
                    input_modal,
                    input_ids=input_ids,
                    modal_start_tokens=modal_start_tokens,
                    modal_end_tokens=modal_end_tokens,
                    attention_mask=attention_mask,
                    token_type_ids=None,
                    modal_token_type_ids=None,
                    position_ids=None,
                    modal_position_ids=None,
                    head_mask=None,
                    inputs_embeds=None,
                    labels=None,
                    return_dict=True,
                )
            else:
                outputs = model(
                    input_modal,
                    input_ids=input_ids,
                    modal_start_tokens=modal_start_tokens,
                    modal_end_tokens=modal_end_tokens,
                    attention_mask=attention_mask,
                    token_type_ids=None,
                    modal_token_type_ids=None,
                    position_ids=None,
                    modal_position_ids=None,
                    head_mask=None,
                    inputs_embeds=None,
                    labels=None, # also set to None for now; originally was Labels
                    return_dict=True,
                )

            if args.use_cvae:
                latents.append(model.image_encoder.latent)
                outputs, mu, logvar, reconstructed_mean, reconstructed_kappa = outputs
                treatments_raw, covariates = get_treatment_and_covariates(batch, args, 6)
                
                # Compute C-VAE loss
                loss_cvae, recon_loss, kl_loss = cvae_loss_fn(reconstructed_mean, reconstructed_kappa, treatments_raw, mu, logvar)
                eval_cvae_loss += loss_cvae.item()
                eval_recon_cvae_loss += recon_loss.item()
                eval_kl_cvae_loss += kl_loss.item()

                treatments, covariates = get_treatment_and_covariates(batch, args, 2)
                
                # Perform PPC for this batch and accumulate results
                batch_ppc = ppc_checker.posterior_predictive_check(
                    model.image_encoder.cvae_encoder, 
                    model.image_encoder.cvae_decoder, 
                    treatments, 
                    covariates,
                    treatments_raw
                )
                # Collect per-batch statistics
                ppc_all_mean_p_values.append(batch_ppc['mean_p_value'])
                ppc_all_p_values.extend(batch_ppc['p_value'].flatten())
                ppc_all_is_valid_samples.extend(batch_ppc['is_valid'].flatten())
                ppc_all_test_stats_true.extend(batch_ppc['test_stat_true'].flatten())
                ppc_batch_count += 1

            #logits = outputs[0]  # model outputs are always tuple in transformers (see doc)
            #tmp_eval_loss = criterion(logits, labels)
            logits = outputs.logits
        #    print ("outputs.logits",logits)

            if args.multiclass:
                #my_loss = FocalLoss()
                # my_loss = torch.nn.MultiLabelSoftMarginLoss(reduction='mean')
                # my_loss = torch.nn.MSELoss()
                my_loss = torch.nn.L1Loss()
                tmp_eval_loss = my_loss(logits, labels)

                                               
                #criterion1 = get_multiclass_criterion(eval_dataset)
                #tmp_eval_loss = criterion1(logits, labels)
            else:
                criterion = get_multiclass_criterion(eval_dataset)
                tmp_eval_loss = criterion(logits, labels)                
                #tmp_eval_loss = outputs.loss
            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
        # Move logits and labels to CPU
        if args.multiclass:
        #    pred = logits.detach().cpu().numpy()
            pred = torch.tensor(logits).to(args.device)
            # with open('Prediction_3img_Beesel_ame0514.txt', 'a') as f:
            #     for line in pred:
            #         f.write('[' + ', '.join([f'{num:.4f}' for num in line]) + ']\n')            

            #pred_max = torch.sigmoid(logits).cpu().detach().numpy().max(axis=1)
            #pred = torch.sigmoid(logits).cpu().detach().numpy() == pred_max
          
        else:            
            #pred = torch.sigmoid(logits).cpu().detach().numpy() > 0.5
            #pred = torch.nn.functional.softmax(logits, dim=1).argmax(dim=1).cpu().detach().numpy() # when dim=1, softmax is applied across columns along that dimension
            pred = torch.nn.functional.softmax(logits, dim=1).argmax(dim=1).cpu().detach().numpy()

        if args.multiclass:
            # out_label_id = labels.detach().cpu().numpy() # when multi-label
            out_label_id = torch.tensor(labels).to(args.device)
        else:
            out_label_id = labels.argmax(dim=1).detach().cpu().numpy() # when single-label
        
        if test:
            attn = torch.stack(outputs.attentions)
            text_end = attn.shape[-1]
            boundaries = {
                # "cls":        (0, 1),
                "RS":         (1, 4),
                "DSM":        (4, 7),
                "NLRS":       (7, 10),
                # "AlphaEarth": (10, 13),
                # "sep":        (13, 14),
                # "text":       (14, text_end),
            }
            next_idx = 10
            if alphaearth:
                boundaries["AlphaEarth"] = (next_idx, next_idx + 3)
                next_idx += 3
            if anysat:
                boundaries["AnySat"] = (next_idx, next_idx + 3)
                next_idx += 3
            if terramind:
                boundaries["TerraMind"] = (next_idx, next_idx + 3)
                next_idx += 3
            boundaries["text"] = (next_idx + 1, text_end)  # +1 for sep token
            
            l_attn = []
            attn_combined = attn.sum(dim=0).mean(dim=2)

            for modality, (start, end) in boundaries.items():
                # incoming: mean over query and key tokens -> (16,)
                score = attn_combined[:, :, start:end].sum(dim=(1, 2))  # (16,)
                l_attn.append(score.cpu())

            l_attn = torch.stack(l_attn, dim=1)
            attentions.append(l_attn)

        preds.append(pred)  # append predictions (default concatenation direction); use axis=0 for vertical stacking
        #print("preds", preds)
        out_label_ids.append(out_label_id)
  #      print("out_label_ids", out_label_ids)
        

               

    eval_loss = eval_loss / nb_eval_steps

    result = {"loss": eval_loss}
    
    if args.multiclass:
        #tgts = np.vstack(out_label_ids)
        tgts = torch.cat(out_label_ids, dim=0)
        
        preds = torch.cat(preds, dim=0)#np.vstack(preds)
        if test:
            attentions = torch.cat(attentions, dim=0)

        if args.use_cvae:
            latents = torch.cat(latents, dim=0)

        tgts_lbm = tgts[:, 0]
        tgts_fys = tgts[:, 1]
        tgts_onv = tgts[:, 2]
        tgts_soc = tgts[:, 3]
        tgts_vrz = tgts[:, 4]
        tgts_won = tgts[:, 5]

        preds_lbm = preds[:, 0]
        preds_fys = preds[:, 1]
        preds_onv = preds[:, 2]
        preds_soc = preds[:, 3]
        preds_vrz = preds[:, 4]
        preds_won = preds[:, 5]        



    #    result["rmse_6"] = mean_squared_error(tgts, preds, squared=False) # targets first, predictions second. If squared=True, mean_squared_error returns MSE instead of RMSE
        result["rmse_6"] = torch.sqrt(torch.mean((tgts - preds) ** 2)) # Alex: (np.sqrt(mean_squared_error(region_obsv['pred'], region_obsv['gt'])))
        # Compute R-squared (goodness of fit)
        mean_tgts = torch.mean(tgts)
        ss_tot = torch.sum((tgts - mean_tgts) ** 2)
        ss_res = torch.sum((tgts - preds) ** 2)
        result["r_squared_6"] = 1 - (ss_res / ss_tot)     
    #r2_score(tgts, preds) # Alex: (linregress(region_obsv['pred'], region_obsv['gt']).rvalue)
        preds_np = preds.cpu().numpy().ravel()
        tgts_np = tgts.cpu().numpy().ravel()
        result["r_Pearson _6"] = linregress(preds_np, tgts_np).rvalue


    #    result["rmse_lbm"] = mean_squared_error(tgts_lbm, preds_lbm, squared=False)# 
        result["rmse_lbm"] = torch.sqrt(torch.mean((tgts_lbm - preds_lbm) ** 2))
        mean_tgts_lbm = torch.mean(tgts_lbm)
        ss_tot_lbm = torch.sum((tgts_lbm - mean_tgts_lbm) ** 2)
        ss_res_lbm = torch.sum((tgts_lbm - preds_lbm) ** 2)
        result["r_squared_lbm"] = 1 - (ss_res_lbm / ss_tot_lbm)
        preds_lbm_np = preds_lbm.cpu().numpy().ravel()
        tgts_lbm_np = tgts_lbm.cpu().numpy().ravel()
        try:
            result["r_Pearson_lbm"] = linregress(preds_lbm_np, tgts_lbm_np).rvalue
        except ValueError:
            result["r_Pearson_lbm"] = float("nan")
        # result["r_Pearson _lbm"] = linregress(preds_lbm_np, tgts_lbm_np).rvalue


        result["rmse_tgts_fys"] = torch.sqrt(torch.mean((tgts_fys - preds_fys) ** 2))#mean_squared_error(tgts_fys, preds_fys, squared=False)# 
        #result["r_squared_fys"] = r2_score(tgts_fys, preds_fys)
        mean_tgts_fys = torch.mean(tgts_fys)
        ss_tot_fys = torch.sum((tgts_fys - mean_tgts_fys) ** 2)
        ss_res_fys = torch.sum((tgts_fys - preds_fys) ** 2)
        result["r_squared_fys"] = 1 - (ss_res_fys / ss_tot_fys)
    
        preds_fys_np = preds_fys.cpu().numpy().ravel()
        tgts_fys_np = tgts_fys.cpu().numpy().ravel()
        result["r_Pearson _fys"] = linregress(preds_fys_np, tgts_fys_np).rvalue

        result["rmse_tgts_onv"] = torch.sqrt(torch.mean((tgts_onv - preds_onv) ** 2))#mean_squared_error(tgts_onv, preds_onv, squared=False)# 
        #result["r_squared_onv"] = r2_score(tgts_onv, preds_onv)
        mean_tgts_onv = torch.mean(tgts_onv)
        ss_tot_onv = torch.sum((tgts_onv - mean_tgts_onv) ** 2)
        ss_res_onv = torch.sum((tgts_onv - preds_onv) ** 2)
        result["r_squared_onv"] = 1 - (ss_res_onv / ss_tot_onv)   
        preds_onv_np = preds_onv.cpu().numpy().ravel()
        tgts_onv_np = tgts_onv.cpu().numpy().ravel()        
        result["r_Pearson _onv"] = linregress(preds_onv_np, tgts_onv_np).rvalue

        result["rmse_tgts_soc"] = torch.sqrt(torch.mean((tgts_soc - preds_soc) ** 2))#mean_squared_error(tgts_soc, preds_soc, squared=False)# 
        #result["r_squared_soc"] = r2_score(tgts_soc, preds_soc)
        mean_tgts_soc = torch.mean(tgts_soc)
        ss_tot_soc = torch.sum((tgts_soc - mean_tgts_soc) ** 2)
        ss_res_soc = torch.sum((tgts_soc - preds_soc) ** 2)
        result["r_squared_soc"] = 1 - (ss_res_soc / ss_tot_soc) 
        preds_soc_np = preds_soc.cpu().numpy().ravel()
        tgts_soc_np = tgts_soc.cpu().numpy().ravel()        
        result["r_Pearson_soc"] = linregress(preds_soc_np, tgts_soc_np).rvalue

        result["rmse_tgts_vrz"] = torch.sqrt(torch.mean((tgts_vrz - preds_vrz) ** 2))#mean_squared_error(tgts_vrz, preds_vrz, squared=False)# 
       # result["r_squared_vrz"] = r2_score(tgts_vrz, preds_vrz)
        mean_tgts_vrz = torch.mean(tgts_vrz)
        ss_tot_vrz = torch.sum((tgts_vrz - mean_tgts_vrz) ** 2)
        ss_res_vrz = torch.sum((tgts_vrz - preds_vrz) ** 2)
        result["r_squared_vrz"] = 1 - (ss_res_vrz / ss_tot_vrz) 
        preds_vrz_np = preds_vrz.cpu().numpy().ravel()
        tgts_vrz_np = tgts_vrz.cpu().numpy().ravel()          
        result["r_Pearson_vrz"] = linregress(preds_vrz_np, tgts_vrz_np).rvalue

        result["rmse_tgts_won"] = torch.sqrt(torch.mean((tgts_won - preds_won) ** 2))#mean_squared_error(tgts_won, preds_won, squared=False)# 
       # result["r_squared_won"] = r2_score(tgts_won, preds_won)
        mean_tgts_won = torch.mean(tgts_won)
        ss_tot_won = torch.sum((tgts_won - mean_tgts_won) ** 2)
        ss_res_won = torch.sum((tgts_won - preds_won) ** 2)
        result["r_squared_won"] = 1 - (ss_res_won / ss_tot_won)
        preds_won_np = preds_won.cpu().numpy().ravel()
        tgts_won_np = tgts_won.cpu().numpy().ravel()  
        result["r_Pearson_won"] = linregress(preds_won.cpu().numpy(), tgts_won.cpu().numpy()).rvalue

        # Extract ground-truth and predicted values for each label
#        tgts_lbm_plot = tgts[:, 0].cpu().numpy()
#        tgts_fys_plot = tgts[:, 1].cpu().numpy()
#        tgts_onv_plot = tgts[:, 2].cpu().numpy()
#        tgts_soc_plot = tgts[:, 3].cpu().numpy()
#        tgts_vrz_plot = tgts[:, 4].cpu().numpy()
#        tgts_won_plot = tgts[:, 5].cpu().numpy()

#        preds_lbm_plot = preds[:, 0].cpu().numpy()
#        preds_fys_plot = preds[:, 1].cpu().numpy()
#        preds_onv_plot = preds[:, 2].cpu().numpy()
#        preds_soc_plot = preds[:, 3].cpu().numpy()
#        preds_vrz_plot = preds[:, 4].cpu().numpy()
#        preds_won_plot = preds[:, 5].cpu().numpy()

        # Create plots
    #    fig, axes = plt.subplots(3, 2, figsize=(15, 15))
    #    plot_density_scatter(tgts_lbm_plot, preds_lbm_plot, 'Livability', axes[0, 0], xlim=(3.4, 4.6), ylim=(3.4, 4.6))
    #    plot_density_scatter(tgts_fys_plot, preds_fys_plot, 'Phy', axes[0, 1], xlim=(-0.3, 0.2), ylim=(-0.3, 0.2))
   #     plot_density_scatter(tgts_onv_plot, preds_onv_plot, 'Nui', axes[1, 0], xlim=(-0.5, 0.2), ylim=(-0.5, 0.2))
   #     plot_density_scatter(tgts_soc_plot, preds_soc_plot, 'Soc', axes[1, 1], xlim=(-0.1, 0.35), ylim=(-0.1, 0.35))
   #     plot_density_scatter(tgts_vrz_plot, preds_vrz_plot, 'Ame', axes[2, 0], xlim=(-0.2, 0.4), ylim=(-0.2, 0.4))
   #     plot_density_scatter(tgts_won_plot, preds_won_plot, 'Hou', axes[2, 1], xlim=(-0.3, 0.4), ylim=(-0.3, 0.4))

#        plt.tight_layout()
#        plt.show()
           
    else:
          
        
        preds = [l for sl in preds for l in sl]

        out_label_ids = [l for sl in out_label_ids for l in sl]

        result["micro_f1"] = f1_score(out_label_ids, preds, average="micro")
#        print ("micro_f1",result["micro_f1"])
        result["macro_f1"] = f1_score(out_label_ids, preds, average="macro")

    # ===== Posterior Predictive Checks for C-VAE =====
    if args.use_cvae and ppc_batch_count > 0:
        logger.info("***** Aggregating Posterior Predictive Checks (PPC) for C-VAE *****")
        
        # Aggregate PPC results across all batches and samples
        ppc_all_p_values = np.array(ppc_all_p_values)
        ppc_all_is_valid_samples = np.array(ppc_all_is_valid_samples)
        ppc_all_test_stats_true = np.array(ppc_all_test_stats_true)
        ppc_all_mean_p_values = np.array(ppc_all_mean_p_values)
        
        # Compute aggregate statistics
        aggregate_mean_p_value = np.mean(ppc_all_p_values)
        aggregate_std_p_value = np.std(ppc_all_p_values)
        num_valid_samples = int(np.sum(ppc_all_is_valid_samples))
        total_samples = len(ppc_all_is_valid_samples)
        sample_validity_rate = num_valid_samples / total_samples if total_samples > 0 else 0.0
        
        # Test statistic distributions
        test_stat_mean = np.mean(ppc_all_test_stats_true)
        test_stat_std = np.std(ppc_all_test_stats_true)
        test_stat_min = np.min(ppc_all_test_stats_true)
        test_stat_max = np.max(ppc_all_test_stats_true)
        
        # Check if model passes validity criterion at aggregate level
        is_valid_aggregate = 0.25 < aggregate_mean_p_value < 0.75
        
        # Store results
        result["ppc_mean_p_value"] = float(aggregate_mean_p_value)
        result["ppc_std_p_value"] = float(aggregate_std_p_value)
        result["ppc_sample_validity_rate"] = float(sample_validity_rate)
        result["ppc_num_valid_samples"] = int(num_valid_samples)
        result["ppc_total_samples"] = int(total_samples)
        result["ppc_is_valid_aggregate"] = float(is_valid_aggregate)
        result["ppc_test_stat_mean"] = float(test_stat_mean)
        result["ppc_test_stat_std"] = float(test_stat_std)
        result["ppc_test_stat_min"] = float(test_stat_min)
        result["ppc_test_stat_max"] = float(test_stat_max)
        result["ppc_batch_count"] = int(ppc_batch_count)
        result["ppc_mean_batch_p_value"] = float(np.mean(ppc_all_mean_p_values))
        result["ppc_std_batch_p_value"] = float(np.std(ppc_all_mean_p_values))
        
        # Log detailed results
        logger.info("  PPC aggregated mean p-value: %.4f ± %.4f", aggregate_mean_p_value, aggregate_std_p_value)
        logger.info("  PPC is valid at aggregate level (0.25 < p < 0.75): %s", is_valid_aggregate)
        logger.info("  PPC sample validity rate: %d / %d (%.2f%%)", num_valid_samples, total_samples, sample_validity_rate * 100)
        logger.info("  PPC test statistic distribution: mean=%.4f, std=%.4f, range=[%.4f, %.4f]", 
                   test_stat_mean, test_stat_std, test_stat_min, test_stat_max)
        logger.info("  PPC batches evaluated: %d", ppc_batch_count)
        
        if not is_valid_aggregate:
            logger.warning("PPC indicates potential model misspecification! (p-value outside [0.25, 0.75])")

    if prefix != "":
        if evaluate and not test:
            output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results_val.txt")
        elif evaluate and test:
            output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results_test.txt")
        else:
            raise ValueError("evaluate must be True.")
                
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results {} *****".format(prefix))
            for key in sorted(result.keys()):
                logger.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))
                # if test:
                #     tb_writer.add_scalar(f'eval_{key}', result[key], nb_eval_steps)
        
        # if test:
        #     tb_writer.close()


    if test:
        if args.use_cvae:
            return result, tgts, preds, latents, attentions
        return result, tgts, preds, attentions
    return result


import torch.nn as nn

class ModalityGradCAMWrapper(nn.Module):
    def __init__(self, model, input_ids, attention_mask, 
                 modal_start_tokens, modal_end_tokens):
        super().__init__()
        self.model = model
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.modal_start_tokens = modal_start_tokens
        self.modal_end_tokens = modal_end_tokens

    def forward(self, full_input_modal):
        """
        full_input_modal: (B, 9 or 9+64, 224, 224) — full input with all modalities
        GradCAM hooks on the target layer will isolate which modality's 
        activations/gradients to use — no need to splice here.
        """
        outputs = self.model(
            full_input_modal,
            input_ids=self.input_ids,
            modal_start_tokens=self.modal_start_tokens,
            modal_end_tokens=self.modal_end_tokens,
            attention_mask=self.attention_mask,
            token_type_ids=None,
            modal_token_type_ids=None,
            position_ids=None,
            modal_position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            return_dict=True,
        )
        if self.model.cvae:
            outputs, *_ = outputs
        return outputs.logits


def get_target_layers(model, modality, last_only=True):
    densenet = model.mmbt.modal_encoder.encoder.model[0]

    if modality in ("RS", "DSM", "NLRS"):
        layers = []
        for block_name in ["denseblock1", "denseblock2", "denseblock3", "denseblock4"]:
            block = getattr(densenet, block_name)
            for layer_name, layer in block.named_children():
                if hasattr(layer, "conv2"):
                    layers.append(layer.conv2)
        return [layers[-1]] if last_only else layers

    elif modality == "AlphaEarth":
        layers = [
            layer for layer in model.mmbt.modal_encoder.encoder.ae_conv
            if isinstance(layer, nn.Conv2d)
        ]
        return [layers[-1]] if last_only else layers

import torch.nn.functional as F

def full_gradcam_regression(wrapped, modality_input, target_layers, modality, target_size=(224, 224)):
    
    modality_order = {"RS": 0, "DSM": 1, "NLRS": 2}
    target_idx = modality_order.get(modality, 0)

    # One activations_list and gradients dict per layer
    all_activations = {i: [] for i in range(len(target_layers))}  # layer_idx -> list of activations
    all_gradients   = {i: {} for i in range(len(target_layers))}  # layer_idx -> {"value": grad}

    handles = []
    for layer_idx, layer in enumerate(target_layers):
        def make_fwd_hook(lidx):
            def fwd_hook(module, input, output):
                call_count = len(all_activations[lidx])
                all_activations[lidx].append(output.detach())
                if call_count == target_idx:
                    output.register_hook(
                        lambda grad, li=lidx: all_gradients[li].update({"value": grad})
                    )
            return fwd_hook
        handles.append(layer.register_forward_hook(make_fwd_hook(layer_idx)))

    modality_input = modality_input.requires_grad_(True)
    logits = wrapped(modality_input)
    num_outputs = logits.shape[1]
    B = modality_input.shape[0]

    cams_per_output = []

    for output_idx in range(num_outputs):
        wrapped.model.zero_grad()

        # Clear all activations and gradients
        for i in range(len(target_layers)):
            all_activations[i].clear()
            all_gradients[i].clear()

        logits = wrapped(modality_input)
        score  = logits[:, output_idx].sum()
        score.backward(retain_graph=True)

        # Sum CAMs from all layers onto a zero array of target_size
        cam_sum = np.zeros((B, target_size[0], target_size[1]), dtype=np.float32)

        for layer_idx in range(len(target_layers)):
            if "value" not in all_gradients[layer_idx]:
                continue
            if len(all_activations[layer_idx]) <= target_idx:
                continue

            act  = all_activations[layer_idx][target_idx]   # (B, C, H, W)
            grad = all_gradients[layer_idx]["value"]         # (B, C, H, W)

            weights = grad.mean(dim=(2, 3), keepdim=True)    # (B, C, 1, 1)
            cam = (weights * act).sum(dim=1, keepdim=True)   # (B, 1, H, W)

            # Interpolate this layer's cam to target_size
            cam = F.interpolate(cam, size=target_size, mode='bilinear', align_corners=False)
            cam = cam.squeeze(1)  # (B, H, W)

            cam_sum += cam.detach().cpu().numpy()  # accumulate onto zero array

        cams_per_output.append(cam_sum)  # (B, 224, 224)

    for h in handles:
        h.remove()

    return np.stack(cams_per_output, axis=1)  # (B, num_outputs, 224, 224)

def run_gradcam(args, model, eval_dataloader, alphaearth):
    """
    Run GradCAM for each modality across the full eval set.
    Returns per-sample GradCAM maps for each modality.
    """
    modalities = ["RS", "DSM", "NLRS"]
    if alphaearth:
        modalities.append("AlphaEarth")

    all_gradcam = {}  # {global_sample_idx: {modality: (num_outputs, H, W)}}

    special_cases = [2560, 10714]

    global_idx = 0  # track position across batches

    for batch in tqdm(eval_dataloader, desc="GradCAM"):
        batch_size = batch[0].shape[0]
        batch_indices = list(range(global_idx, global_idx + batch_size))
    
        # Check if any special cases fall in this batch
        special_in_batch = [i for i in special_cases if i in batch_indices]
        
        if not special_in_batch:
            global_idx += batch_size
            continue  # skip batch entirely

        # Get local indices within this batch
        local_indices = [batch_indices.index(i) for i in special_in_batch]

        batch = tuple(t.to(args.device) for t in batch)
        input_ids          = batch[0]
        attention_mask     = batch[1]
        input_modal        = batch[2]   # (B, 9 or 9+64, 224, 224)
        modal_start_tokens = batch[3]
        modal_end_tokens   = batch[4]
        # labels             = batch[5]

        # Slice only the relevant samples from the batch
        input_ids_sub          = input_ids[local_indices]
        attention_mask_sub     = attention_mask[local_indices]
        input_modal_sub        = input_modal[local_indices]
        modal_start_tokens_sub = modal_start_tokens[local_indices]
        modal_end_tokens_sub   = modal_end_tokens[local_indices]

        for modality in modalities:
            # 1. Wrap model for this modality
            wrapped = ModalityGradCAMWrapper(
                model=model,
                input_ids=input_ids_sub,
                attention_mask=attention_mask_sub,
                modal_start_tokens=modal_start_tokens_sub,
                modal_end_tokens=modal_end_tokens_sub,
            )

            # 2. Get target layer
            target_layers = get_target_layers(model, modality)

            # 3. Slice out just this modality's channels as the "input" for GradCAM
            # c_start, c_end = wrapped.channel_map[modality]
            # if modality == "AlphaEarth":
            #     modality_input = input_modal[:, c_start:c_end, :50, :50]  # (B, 64, 50, 50)
            # else:
            #     modality_input = input_modal[:, c_start:c_end, :, :]      # (B, 3, 224, 224)

            cam = full_gradcam_regression(
                wrapped=wrapped,
                modality_input=input_modal_sub.requires_grad_(True),
                modality=modality,
                target_layers=target_layers,
                target_size=(50, 50) if modality == "AlphaEarth" else (224, 224),
            )
            
            # Store with global index as key
            for i, global_sample_idx in enumerate(special_in_batch):
                if global_sample_idx not in all_gradcam:
                    all_gradcam[global_sample_idx] = {}
                all_gradcam[global_sample_idx][modality] = cam[i]  # (num_outputs, H, W)
        global_idx += batch_size
        

    return all_gradcam

def gradcam(args, model, tokenizer, alphaearth, alphazero, anysat=False, terramind=False, anyzero=False, terrazero=False, evaluate=True, test=False, prefix=""):
    """
    Evaluate the model with optional C-VAE confounder reconstruction and posterior predictive checks.
    
    Args:
        args: arguments
        model: MMBT model
        tokenizer: text tokenizer
        alphaearth: whether to use alphaearth data
        evaluate: whether this is eval (vs test)
        test: whether this is test set
        prefix: prefix for output files
    """
    model.eval()
    eval_output_dir = args.output_dir
    eval_dataset = load_examples(tokenizer, args, alphaearth=alphaearth, alphazero=alphazero, anysat=anysat, terramind=terramind, anyzero=anyzero, terrazero=terrazero, evaluate=evaluate, test=test)

    if not os.path.exists(eval_output_dir):
        os.makedirs(eval_output_dir)

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, collate_fn=collate_fn
    )

    gradcam_maps = run_gradcam(args, model, eval_dataloader, alphaearth)

    return gradcam_maps

