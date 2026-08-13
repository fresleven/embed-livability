import sys
import os
import argparse
import glob
import logging
import warnings
import torch
import json
import numpy as np

# Make sibling modules (utils, textBert_utils, MMBT_liva, ...) importable
# regardless of the working directory the job is launched from.
py_file_location = os.path.dirname(os.path.abspath(__file__))
sys.path.append(py_file_location)

from textBert_utils import set_seed
from MMBT_liva.image_liva import ImageEncoderDenseNet
from MMBT_liva.mmbt_config_liva import MMBTConfig
from MMBT_liva.mmbt_liva import MMBTForClassification
from MMBT_liva.mmbt_utils_liva_0318 import load_examples, get_multiclass_labels, get_labels

from transformers import WEIGHTS_NAME, AutoConfig, AutoModel, AutoTokenizer

from utils import train, evaluate

# Ignore all warnings
warnings.filterwarnings("ignore")
# Or ignore a specific category of warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# File paths for train, val, test data
possible_suffixes = ["", "_NULL_RS", "_NULL_DSM", "_NULL_GIU", "_noPOI", "_NULL", "_only_RS", "_only_DSM", "_only_GIU", "_only_POI"]

# Model specification (constants only; all flags come from CLI args below)
do_multiclass = True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default="data_livability/json",
        type=str,
        help="The input data dir. Should contain the .jsonl files.",
    )
    parser.add_argument(
        "--model_name",
        default="bert-base-multilingual-uncased", 
        type=str,
        help="model identifier from huggingface.co/models",
    )
    # If the input language is Chinese, change this model choice accordingly.
    # Examples: 'bert-base-uncased' for lowercase English; 'bert-base-chinese' for Chinese.
    parser.add_argument(
        "--output_dir",
        default="0613_livability_4M_6aspects_length400_selected", #   mmbt_output_findings_10epochs_n
        type=str,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

        
    parser.add_argument(
        "--config_name", default="bert-base-multilingual-uncased", type=str, help="Pretrained config name if not the same as model_name"
    )
    parser.add_argument(
        "--tokenizer_name",
        default="bert-base-multilingual-uncased",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )

    parser.add_argument("--train_batch_size", default=16, type=int, help="Batch size for training.") # changed from 32 to 16; batch size notes
    parser.add_argument(
        "--eval_batch_size", default=16, type=int, help="Batch size for evaluation." # changed from 32 to 16; set to 1 for attention visualization
    )
    parser.add_argument(
        "--max_seq_length",
        default=400,#50,  # originally 300, max 512
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--num_image_embeds", default=9, type=int, help="Number of Image Embeddings from the Image Encoder" # 3chanel *3 images
    ) # B*N*1024 (batch * num_image_embeds * 1024)
    parser.add_argument("--do_train", action=argparse.BooleanOptionalAction, default=True, help="Whether to run training.")
    parser.add_argument("--do_eval", action=argparse.BooleanOptionalAction, default=True, help="Whether to run eval on the test set.")
    parser.add_argument(
        "--evaluate_during_training", action=argparse.BooleanOptionalAction, default=True, help="Run evaluation during training at each logging step."
    )


    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument("--learning_rate", default=5e-05, type=float, help="The initial learning rate for Adam.") # previously default was 5e-5 (0.00005)
    parser.add_argument("--weight_decay", default=0.1, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs", default=12, type=float, help="Total number of training epochs to perform." # Huggingface default is 3; previous experiments used 10
    )
    parser.add_argument("--patience", default=5, type=int, help="Patience for Early Stopping.")
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.") # When num_warmup_steps is 0 the learning rate has no warmup phase and only decays.

    parser.add_argument("--logging_steps", type=int, default=50, help="Log every X updates steps.") # previously 25
    parser.add_argument("--save_steps", type=int, default=50, help="Save checkpoint every X updates steps.") # previously 25
    parser.add_argument(
        "--eval_all_checkpoints",
        action=argparse.BooleanOptionalAction, default=True,
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )

    parser.add_argument("--num_workers", type=int, default=8, help="number of worker threads for dataloading")

    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")

    # C-VAE specific hyperparameters
    parser.add_argument("--use_cvae", action=argparse.BooleanOptionalAction, default=False, help="Whether to use C-VAE for confounder reconstruction")
    parser.add_argument("--cvae_latent_d", default=3, type=int, help="Depth/channels of latent space for C-VAE")
    parser.add_argument("--cvae_encoder_base_channels", default=64, type=int, help="Base channels for C-VAE encoder")
    parser.add_argument("--cvae_encoder_depth", default=3, type=int, help="Number of conv layers in C-VAE encoder")
    parser.add_argument("--cvae_decoder_base_channels", default=64, type=int, help="Base channels for C-VAE decoder")
    parser.add_argument("--cvae_decoder_depth", default=3, type=int, help="Number of conv layers in C-VAE decoder")
    parser.add_argument("--cvae_dropout", default=0.1, type=float, help="Dropout rate for C-VAE")
    parser.add_argument("--cvae_kl_weight", default=1, type=float, help="Weight for KL divergence in C-VAE (beta)")
    parser.add_argument("--cvae_kl_warmup_epochs", default=3, type=int, help="Number of epochs to warm up KL weight")
    parser.add_argument("--cvae_treatment_source", default="image", type=str, help="Source of treatment: 'alphaearth' or 'image'")
    parser.add_argument("--cvae_covariate_source", default="alphaearth", type=str, help="Source of covariates: 'alphaearth' or 'image'")
    parser.add_argument("--var", action=argparse.BooleanOptionalAction, default=True, help="Whether to use variance in treatment modeling (Gaussian or vMF)")
    parser.add_argument("--cvae_latent_hw", default=50, type=int, help="Spatial dimensions of latent space for C-VAE")
    parser.add_argument("--text_conditioning", action=argparse.BooleanOptionalAction, default=False, help="Whether to condition C-VAE on text embeddings")
    parser.add_argument("--text_embedding_dim", default=768, type=int, help="Dimension of text embeddings")

    # Embedding source flags
    parser.add_argument("--alphaearth", action=argparse.BooleanOptionalAction, default=False, help="Use AlphaEarth embeddings")
    parser.add_argument("--anysat", action=argparse.BooleanOptionalAction, default=False, help="Use AnySat embeddings")
    parser.add_argument("--terramind", action=argparse.BooleanOptionalAction, default=False, help="Use TerraMind embeddings")
    parser.add_argument("--alphazero", action=argparse.BooleanOptionalAction, default=False, help="Zero out AlphaEarth embeddings for ablation")
    parser.add_argument("--terrazero", action=argparse.BooleanOptionalAction, default=False, help="Zero out TerraMind embeddings for ablation")
    parser.add_argument("--anyzero", action=argparse.BooleanOptionalAction, default=False, help="Zero out AnySat embeddings for ablation")

    # Training control flags
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False, help="Resume from latest checkpoint in output_dir")
    parser.add_argument("--final_checkpoint", action=argparse.BooleanOptionalAction, default=True, help="Evaluate only final checkpoint (not all)")

    # Data split selection
    parser.add_argument(
        "--suffix_index", type=int, default=0,
        choices=list(range(len(possible_suffixes))),
        help="Index into possible_suffixes: 0='' 1=_NULL_RS 2=_NULL_DSM 3=_NULL_GIU 4=_noPOI 5=_NULL 6=_only_RS 7=_only_DSM 8=_only_GIU 9=_only_POI",
    )
    parser.add_argument(
        "--test_suffix_index", type=int, default=-1,
        help="Override test file suffix independently of suffix_index (-1 = use suffix_index)",
    )

    args = parser.parse_args()

    # Derive local variables from parsed args
    alphaearth = args.alphaearth
    anysat = args.anysat
    terramind = args.terramind
    alphazero = args.alphazero
    terrazero = args.terrazero
    anyzero = args.anyzero
    use_cvae = args.use_cvae
    resume = args.resume
    final_checkpoint = args.final_checkpoint

    suffix = possible_suffixes[args.suffix_index]
    train_file = "Livability_train_0320" + suffix + ".json"
    val_file = "Livability_eval_0320" + suffix + ".json"
    test_suffix = possible_suffixes[args.test_suffix_index] if args.test_suffix_index >= 0 else suffix
    test_file = "Livability_test_0320" + test_suffix + ".json"
    args.train_file = train_file
    args.val_file = val_file
    args.test_file = test_file

    # Setup CUDA, GPU & distributed training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    args.device = device

    # for multiclass labeling
    args.multiclass = do_multiclass # (was modified in prior runs as needed)
    args.eval_all_checkpoints = not final_checkpoint

    # change output dir
    args.output_dir = "ckpt_livability"
    if alphaearth:
        args.output_dir = args.output_dir + "_alphaearth"
    if anysat:
        args.output_dir = args.output_dir + "_anysat"
    if terramind:
        args.output_dir = args.output_dir + "_terramind"
    if args.use_cvae:
        args.output_dir = args.output_dir + "_cvae"
        if args.cvae_treatment_source == "alphaearth":
            args.output_dir = args.output_dir + "_ae-treat-mid"
        else:
            args.output_dir = args.output_dir + "_im-treat-mid"
        
        if not args.var:
            args.output_dir = args.output_dir + "_nolog"
        args.output_dir += "_" + str(args.cvae_kl_weight)
    if "NULL" in train_file or "noPOI" in train_file:
        args.output_dir += "_" + train_file.split("0320_", 1)[1].rsplit(".json", 1)[0]
    if args.seed != 42:
        args.output_dir += "_seed" + str(args.seed)
    print("Output dir: ", args.output_dir)
    if alphazero:
        print("Testing with all satellite embeddings zeroed out (ablation)")

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name,
        do_lower_case=True,
        cache_dir=None,
    )
    train_dataset = load_examples(tokenizer, args, alphaearth=alphaearth or args.use_cvae, alphazero=alphazero, anysat=anysat, terramind=terramind, evaluate=False)

    print (train_dataset[0].keys())
    print(train_dataset[0]['image'].shape)

    # Setup logging
    logger = logging.getLogger(__name__)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO)
    # Also log to file
    file_handler = logging.FileHandler(os.path.join(args.output_dir, f"{os.path.splitext(args.train_file)[0]}_logging.txt"))
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s -   %(message)s"))
    logger.addHandler(file_handler)
    logger.warning("device: %s, n_gpu: %s",
            args.device,
            args.n_gpu
    )
    # Set the verbosity to info of the Transformers logger (on main process only):

    # Set seed
    set_seed(args)

    # Setup model
    if args.multiclass:
        labels = get_multiclass_labels()
        num_labels = len(labels)
    else:
        labels = get_labels()
        num_labels = len(labels)
    transformer_config = AutoConfig.from_pretrained(args.config_name if args.config_name else args.model_name, num_labels=num_labels)
    tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_name if args.tokenizer_name else args.model_name,
            do_lower_case=True,
            cache_dir=None,
        )
    transformer = AutoModel.from_pretrained(args.model_name, config=transformer_config, cache_dir=None)
    img_encoder = ImageEncoderDenseNet(
        num_image_embeds=args.num_image_embeds,
        alphaearth=alphaearth,
        anysat=anysat,
        terramind=terramind,
        cvae=args.use_cvae,
        cvae_latent_d=args.cvae_latent_d,
        cvae_encoder_base_channels=args.cvae_encoder_base_channels,
        cvae_encoder_depth=args.cvae_encoder_depth,
        cvae_decoder_base_channels=args.cvae_decoder_base_channels,
        cvae_decoder_depth=args.cvae_decoder_depth,
        cvae_kl_weight=args.cvae_kl_weight,
        cvae_treatment_source=args.cvae_treatment_source,
        cvae_covariate_source=args.cvae_covariate_source,
        var=args.var,
        cvae_latent_hw=args.cvae_latent_hw,
        text_embedding_dim=args.text_embedding_dim if args.text_conditioning else None,
    )
    multimodal_config = MMBTConfig(transformer, img_encoder, num_labels=num_labels, modal_hidden_size=1024)
    model = MMBTForClassification(transformer_config, multimodal_config, cvae=args.use_cvae)

    model.to(args.device)

    logger.info(f"Training/evaluation parameters: {args}")

    # Training
    if args.do_train:
        train_dataset = load_examples(tokenizer, args, alphaearth=alphaearth or use_cvae, alphazero=alphazero, anysat=anysat, terramind=terramind, evaluate=False)

        # Resume from latest checkpoint if requested
        resume_from_step = 0
        resume_checkpoint_dir = None
        if resume:
            existing = sorted(glob.glob(os.path.join(args.output_dir, "checkpoint-[0-9]*")))
            existing = [c for c in existing if os.path.basename(c).split("-")[-1].isdigit()]
            if existing:
                resume_checkpoint_dir = max(existing, key=lambda c: int(os.path.basename(c).split("-")[-1]))
                resume_from_step = int(os.path.basename(resume_checkpoint_dir).split("-")[-1])
                logger.info("Resuming from checkpoint %s (step %d)", resume_checkpoint_dir, resume_from_step)
                model.load_state_dict(torch.load(os.path.join(resume_checkpoint_dir, WEIGHTS_NAME), map_location=args.device))
                model.to(args.device)
            else:
                logger.info("No checkpoint found in %s, starting from scratch.", args.output_dir)

        global_step, tr_loss = train(args, train_dataset, model, tokenizer, alphaearth, anysat=anysat, terramind=terramind, resume_from_step=resume_from_step, resume_checkpoint_dir=resume_checkpoint_dir)
        logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)
        
        # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
        logger.info("Saving model checkpoint to %s", args.output_dir)
        # Save a trained model, confitopguration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()
        model_to_save = (model.module if hasattr(model, "module") else model)  # Take care of distributed/parallel training
        torch.save(model_to_save.state_dict(), os.path.join(args.output_dir, "checkpoint-final",WEIGHTS_NAME))
        tokenizer.save_pretrained(args.output_dir)
        transformer_config.save_pretrained(args.output_dir)
        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        # Load a trained model and vocabulary that you have fine-tuned
        model = MMBTForClassification(transformer_config, multimodal_config, cvae=args.use_cvae)
        model.load_state_dict(torch.load(os.path.join(args.output_dir, "checkpoint-final", WEIGHTS_NAME)))
        tokenizer = AutoTokenizer.from_pretrained(args.output_dir)
        model.to(args.device)
        logger.info("***** Training Finished *****")  

    # Test Evaluation
    results = {}
    print("=== Starting test evaluation ===")
    print(f"do_eval = {args.do_eval}")
    if args.do_eval:
        checkpoints = [os.path.join(args.output_dir, "checkpoint-final")]
        if args.eval_all_checkpoints:
            checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/pytorch_model.bin", recursive=True)))
            # recursive=False because otherwise the parent diretory gets included
            # which is not what we want; only subdirectories
        checkpoints = [os.path.join(args.output_dir, "checkpoint-final")] if not args.eval_all_checkpoints else checkpoints
        
        logger.info("Evaluate the following checkpoints: %s", checkpoints)
        print(f"Found {len(checkpoints)} checkpoints: {checkpoints}")

        all_results = {}

        for checkpoint in checkpoints:
            validation_path = os.path.join(checkpoint, 'eval_results_val.txt')
            if os.path.exists(validation_path):
                with open(validation_path, "r") as f:
                    results = json.load(f)
                all_results[checkpoint] = results

        print("=== Evaluation results ===")
        print(f"Number of results: {len(all_results)}")

        # Find checkpoint with lowest validation RMSE
        best_checkpoint = None
        best_value = float('inf')
        compare_key = "eval_rmse_lbm"

        for checkpoint, results in all_results.items():
            for key, value in results.items():
                if compare_key in key:
                    if value < best_value:
                        best_value = value
                        best_checkpoint = checkpoint

        if best_checkpoint:
            print(f"\n=== Best validation checkpoint ===")
            print(f"Lowest validation RMSE: {best_value:.6f}")
            print(f"Checkpoint path: {best_checkpoint}")
        elif args.eval_all_checkpoints:
            raise ValueError("No valid checkpoint found during validation evaluation.")
        else:
            # No val results — use checkpoint-final directly
            best_checkpoint = checkpoints[0]
            print(f"No val results found, using checkpoint directly: {best_checkpoint}")

        logger.info("Using checkpoint for test evaluation: %s", best_checkpoint)
        print(f"Using checkpoint for test evaluation: {best_checkpoint}")

        checkpoint = best_checkpoint
        global_step = checkpoint.split("-")[-1] if len(checkpoints) > 1 else ""
        print ("global_step",global_step)
        prefix = checkpoint.split("/")[-1] if checkpoint.find("checkpoint") != -1 else ""
        print(f"Loading model: {checkpoint}")
        model = MMBTForClassification(transformer_config, multimodal_config, cvae=args.use_cvae)
        model_path = os.path.join(checkpoint, 'pytorch_model.bin')
        print(f"Loading model file: {model_path}")
        model.load_state_dict(torch.load(model_path))
        model.to(args.device)
        print("Starting evaluation...")
        if args.use_cvae:
            results, tgts, preds, latents, attentions = evaluate(args, model, tokenizer, alphaearth=alphaearth, alphazero=alphazero, anysat=anysat, terramind=terramind, anyzero=anyzero, terrazero=terrazero, evaluate=True, test=True, prefix=prefix)
        else:
            results, tgts, preds, attentions = evaluate(args, model, tokenizer, alphaearth=alphaearth, alphazero=alphazero, anysat=anysat, terramind=terramind, anyzero=anyzero, terrazero=terrazero, evaluate=True, test=True, prefix=prefix)
        print(results.keys())

        results_serializable = {key: float(value.cpu().numpy()) if isinstance(value, torch.Tensor) else value for key, value in results.items()}
        zero_suffix = ""
        if alphazero: zero_suffix += "_alphazero"
        if terrazero: zero_suffix += "_terrazero"
        if anyzero: zero_suffix += "_anyzero"
        if zero_suffix:
            args.test_file = args.test_file.replace(".json", zero_suffix + ".json")
        if args.eval_all_checkpoints:
            test_file = os.path.join(args.output_dir, f"{os.path.splitext(args.test_file)[0]}_eval_results_test.txt")
            preds_file = os.path.join(args.output_dir, f"{os.path.splitext(args.test_file)[0]}_preds_test.npy")
            tgts_file = os.path.join(args.output_dir, f"{os.path.splitext(args.test_file)[0]}_tgts_test.npy")
            attentions_file = os.path.join(args.output_dir, f"{os.path.splitext(args.test_file)[0]}_attentions_test.npy")
            if args.use_cvae:
                latents_file = os.path.join(args.output_dir, f"{os.path.splitext(args.test_file)[0]}_latents_test.npy")
        else:
            test_file = os.path.join(args.output_dir, f"{os.path.splitext(args.test_file)[0]}_eval_results_test_final.txt")
            preds_file = os.path.join(args.output_dir, f"{os.path.splitext(args.test_file)[0]}_preds_test_final.npy")
            tgts_file = os.path.join(args.output_dir, f"{os.path.splitext(args.test_file)[0]}_tgts_test_final.npy")
            attentions_file = os.path.join(args.output_dir, f"{os.path.splitext(args.test_file)[0]}_attentions_test_final.npy")
            if args.use_cvae:
                latents_file = os.path.join(args.output_dir, f"{os.path.splitext(args.test_file)[0]}_latents_test_final.npy")
        with open(test_file, "w") as f:
            json.dump(results_serializable, f, indent=2)
        # save targets and predictions

        np.save(preds_file, preds.cpu().numpy())
        np.save(tgts_file, tgts.cpu().numpy())
        np.save(attentions_file, attentions)
        if args.use_cvae:
            np.save(latents_file, latents.cpu().numpy())
