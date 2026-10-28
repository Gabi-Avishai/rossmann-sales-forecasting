"""Entity-embedding MLP for the Rossmann project, extracted for reuse.

Phase 4 (`rossmann_nn_sota.ipynb`) defines this architecture inline; the code here is
the same model, factored out so later phases can retrain it on a different split
without copy-pasting the architecture. Kept deliberately identical to Phase 4's
version so results stay comparable.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import OrdinalEncoder

NN_CATEGORICAL_COLS = ["Store", "DayOfWeek", "Month", "StoreType", "Assortment", "StateHoliday"]
EMB_DIMS = {"Store": 20, "DayOfWeek": 4, "Month": 6, "StoreType": 3, "Assortment": 2, "StateHoliday": 2}


def rmspe(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    pct_err = (y_true[mask] - y_pred[mask]) / y_true[mask]
    return float(np.sqrt(np.mean(pct_err ** 2)))


def encode_categoricals(fit_df, apply_dfs, cat_cols=NN_CATEGORICAL_COLS):
    """Ordinal-encode categoricals to contiguous 0-indexed codes for nn.Embedding,
    fitting only on `fit_df`. Unknown categories land in one extra per-column slot
    (index == len(categories)), so embedding tables get cardinality len(categories)+1."""
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, dtype=np.int64)
    fit_codes = encoder.fit_transform(fit_df[cat_cols])
    unknown_slots = np.array([len(cats) for cats in encoder.categories_])
    cardinalities = [len(cats) + 1 for cats in encoder.categories_]

    fit_codes = np.where(fit_codes == -1, unknown_slots, fit_codes)
    applied = []
    for df in apply_dfs:
        codes = encoder.transform(df[cat_cols])
        applied.append(np.where(codes == -1, unknown_slots, codes))
    return encoder, cardinalities, fit_codes, applied


class RossmannDataset(Dataset):
    def __init__(self, cat_codes, num_feats, targets):
        self.cat_codes = torch.as_tensor(cat_codes, dtype=torch.long)
        self.num_feats = torch.as_tensor(num_feats, dtype=torch.float32)
        self.targets = torch.as_tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.cat_codes[idx], self.num_feats[idx], self.targets[idx]


class EmbeddingMLP(nn.Module):
    def __init__(self, cardinalities, cat_cols, emb_dims, n_numeric):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(card, emb_dims[col]) for card, col in zip(cardinalities, cat_cols)
        ])
        emb_total = sum(emb_dims[col] for col in cat_cols)
        self.num_bn = nn.BatchNorm1d(n_numeric)
        in_features = emb_total + n_numeric
        self.mlp = nn.Sequential(
            nn.Linear(in_features, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, cat_codes, num_feats):
        embedded = [emb(cat_codes[:, i]) for i, emb in enumerate(self.embeddings)]
        x = torch.cat(embedded + [self.num_bn(num_feats)], dim=1)
        return self.mlp(x).squeeze(-1)


def predict_in_batches(model, cat_arr, num_arr, device, batch_size=8192):
    model.eval()
    cat_t = torch.as_tensor(cat_arr, dtype=torch.long)
    num_t = torch.as_tensor(num_arr, dtype=torch.float32)
    preds = []
    with torch.no_grad():
        for i in range(0, len(cat_t), batch_size):
            cb = cat_t[i:i + batch_size].to(device)
            nb = num_t[i:i + batch_size].to(device)
            preds.append(model(cb, nb).cpu().numpy())
    return np.concatenate(preds)


def eval_rmspe(model, cat_arr, num_arr, sales, open_flag, device):
    pred_log = predict_in_batches(model, cat_arr, num_arr, device)
    pred = np.clip(np.expm1(pred_log), 0, None)
    pred = np.where(open_flag == 0, 0, pred)
    return rmspe(sales, pred)


def train_nn(cardinalities, cat_cols, emb_dims, n_numeric,
             train_cat, train_num, train_y_log, train_sales, train_open,
             val_cat, val_num, val_sales, val_open, device,
             epochs=40, batch_size=1024, lr=1e-3, weight_decay=1e-5, seed=0, verbose=True):
    """Trains for a fixed epoch budget, returning the best-validation-RMSPE checkpoint
    rather than the final epoch's weights -- mini-batch SGD's per-epoch validation curve
    is too noisy to read a single epoch count off by eye."""
    torch.manual_seed(seed)
    model = EmbeddingMLP(cardinalities, cat_cols, emb_dims, n_numeric).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    train_ds = RossmannDataset(train_cat, train_num, train_y_log)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    history = []
    best_val_rmspe = float("inf")
    best_state = None
    best_epoch = None
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for cat_b, num_b, y_b in train_loader:
            cat_b, num_b, y_b = cat_b.to(device), num_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            pred = model(cat_b, num_b)
            loss = nn.functional.mse_loss(pred, y_b)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y_b)
        train_loss = total_loss / len(train_ds)

        train_r = eval_rmspe(model, train_cat, train_num, train_sales, train_open, device)
        val_r = eval_rmspe(model, val_cat, val_num, val_sales, val_open, device)
        scheduler.step(val_r)
        if val_r < best_val_rmspe:
            best_val_rmspe, best_epoch = val_r, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        history.append({"epoch": epoch, "train_loss_logmse": train_loss,
                        "train_rmspe": train_r, "val_rmspe": val_r})
        if verbose:
            print(f"epoch {epoch:>3}: log-MSE = {train_loss:.5f} | train RMSPE = {train_r:.4f} "
                  f"| val RMSPE = {val_r:.4f}", flush=True)

    model.load_state_dict(best_state)
    if verbose:
        print(f"restored best checkpoint: epoch {best_epoch}, val RMSPE = {best_val_rmspe:.4f}")
    return model, pd.DataFrame(history), best_epoch
