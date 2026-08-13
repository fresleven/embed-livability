"""
Embedding-only probe baseline: predicts livability scores from a single
pooled foundation-model embedding vector alone (AlphaEarth, AnySat, or
TerraMind), with no RS/DSM/NLRS/POI inputs and no DenseNet/BERT/Transformer
fusion. This measures how much signal lives in the embedding by itself,
as a floor against which the full multimodal TMTMR+embedding models
(see train.py) can be compared.

Usage:
    python probe_baseline.py --embedding alphaearth --seed 42
    python probe_baseline.py --embedding anysat --seed 43 --hidden_dim 256
    python probe_baseline.py --embedding terramind --seed 44 --hidden_dim 0  # linear probe

Output layout mirrors train.py so aggregate_seeds.py can read it directly:
    ckpt_probe_<embedding>[_seed<N>]/Livability_test_0320_eval_results_test_final.txt
"""
import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data_livability")
JSONL_DATA_DIR = os.path.join(DATA_DIR, "json")
EMBED_DIRS = {
    "alphaearth": os.path.join(DATA_DIR, "alphaearth"),
    "anysat": os.path.join(DATA_DIR, "anysat"),
    "terramind": os.path.join(DATA_DIR, "terramind"),
}
LABEL_KEYS = ["lbm", "fys", "onv", "soc", "vrz", "won"]  # LIV, PHY, NUI, SOC, AME, HOU
METRIC_NAMES = ["rmse_lbm", "rmse_tgts_fys", "rmse_tgts_onv", "rmse_tgts_soc", "rmse_tgts_vrz", "rmse_tgts_won"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def embedding_path(record, embedding_dir):
    # Same filename convention as MMBT_liva/mmbt_utils_liva_0318.py: swap the
    # modality prefix in whichever image filename is valid (not "NULL.tif")
    # for the embedding prefix, and swap ".tif" for "_<year>.npz".
    candidates = [(record["img"], "RS"), (record["giu"], "GIU"), (record["dsm"], "DSM")]
    valid = [(fname, prefix) for fname, prefix in candidates if fname != "NULL.tif"]
    if not valid:
        ref = record.get("ref_img")
        if ref is None:
            raise ValueError(f"All images are NULL.tif for record {record.get('id')} and no ref_img present.")
        valid = [(ref, "RS")]
    fname, prefix = valid[0]
    embed_fname = fname.replace(prefix, os.path.basename(embedding_dir)).replace(".tif", "_2020.npz")
    return os.path.join(embedding_dir, embed_fname)


class PooledEmbeddingDataset(Dataset):
    def __init__(self, jsonl_path, embedding_dir):
        self.embedding_dir = embedding_dir
        with open(jsonl_path) as f:
            self.data = [json.loads(line) for line in f if line.strip()]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        record = self.data[idx]
        arr = np.load(embedding_path(record, self.embedding_dir))["image_data"]  # (C, H, W)
        pooled = arr.mean(axis=(1, 2)).astype(np.float32)  # (C,)
        label = np.array([float(record[k]) for k in LABEL_KEYS], dtype=np.float32)
        return torch.from_numpy(pooled), torch.from_numpy(label)


def build_model(input_dim, hidden_dim, dropout=0.1):
    if hidden_dim <= 0:
        return nn.Linear(input_dim, len(LABEL_KEYS))
    return nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, len(LABEL_KEYS)),
    )


def evaluate(model, loader, device):
    model.eval()
    preds, tgts = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds.append(model(x).cpu())
            tgts.append(y.cpu())
    preds = torch.cat(preds)
    tgts = torch.cat(tgts)
    result = {}
    rmse_all = torch.sqrt(torch.mean((preds - tgts) ** 2))
    result["rmse_6"] = float(rmse_all)
    for i, metric_name in enumerate(METRIC_NAMES):
        result[metric_name] = float(torch.sqrt(torch.mean((preds[:, i] - tgts[:, i]) ** 2)))
    mae_lbm = torch.mean(torch.abs(preds[:, 0] - tgts[:, 0]))
    result["mae_lbm"] = float(mae_lbm)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding", required=True, choices=list(EMBED_DIRS.keys()))
    parser.add_argument("--suffix", default="", help="Data suffix, e.g. '' for full data.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden_dim", type=int, default=256, help="0 = pure linear probe.")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--patience", type=int, default=8)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    embedding_dir = EMBED_DIRS[args.embedding]
    train_ds = PooledEmbeddingDataset(os.path.join(JSONL_DATA_DIR, f"Livability_train_0320{args.suffix}.json"), embedding_dir)
    val_ds = PooledEmbeddingDataset(os.path.join(JSONL_DATA_DIR, f"Livability_eval_0320{args.suffix}.json"), embedding_dir)
    test_ds = PooledEmbeddingDataset(os.path.join(JSONL_DATA_DIR, f"Livability_test_0320{args.suffix}.json"), embedding_dir)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    input_dim = train_ds[0][0].shape[0]
    model = build_model(input_dim, args.hidden_dim, args.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.L1Loss()  # multi-task MAE, consistent with the main paper's loss

    output_dir = f"ckpt_probe_{args.embedding}"
    if args.suffix:
        output_dir += "_" + args.suffix.strip("_")
    if args.seed != 42:
        output_dir += f"_seed{args.seed}"
    os.makedirs(output_dir, exist_ok=True)

    best_val_rmse = float("inf")
    best_state = None
    epochs_since_improve = 0

    for epoch in range(args.epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            optimizer.step()

        val_result = evaluate(model, val_loader, device)
        val_rmse = val_result["rmse_lbm"]
        print(f"epoch {epoch}: val rmse_lbm={val_rmse:.5f}")
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= args.patience:
                print(f"Early stopping at epoch {epoch}")
                break

    model.load_state_dict(best_state)
    test_result = evaluate(model, test_loader, device)
    print("Test results:", json.dumps(test_result, indent=2))

    out_file = os.path.join(output_dir, f"Livability_test_0320{args.suffix}_eval_results_test_final.txt")
    with open(out_file, "w") as f:
        json.dump(test_result, f, indent=2)
    print(f"Wrote {out_file}")


if __name__ == "__main__":
    main()
