import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.BatchNorm1d(dim)
        self.linear1 = nn.Linear(dim, dim)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(dim, dim)
        self.norm2 = nn.BatchNorm1d(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.linear1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.norm2(x)
        x = self.act(x)
        return x + residual


def _augment(X):
    """Apply feature engineering to raw features."""
    elevation = X[:, 0:1]
    slope = X[:, 2:3]
    wild_interactions = elevation * X[:, 10:14]
    top_soil_indices = [42, 36, 45, 46, 23, 17, 15, 16, 30, 19, 43, 26, 51, 52, 53, 25, 35, 49]
    soil_interactions = elevation * X[:, top_soil_indices]
    slope_soil = slope * X[:, [23, 17, 15, 19, 16, 30]]
    elev_sq = elevation ** 2
    elev_cu = elevation ** 3
    return np.hstack([X, wild_interactions, soil_interactions, slope_soil, elev_sq, elev_cu]).astype(np.float32)


def fit(X_train, y_train, *, seed, device, time_budget_s, log):
    start = time.time()
    n_classes = int(y_train.max()) + 1
    n_total = len(X_train)

    # Proper stratified split: hold out 50K for validation
    n_val = 50000
    rng = np.random.RandomState(seed)
    # Stratified sampling
    val_idx = []
    for c in range(n_classes):
        class_idx = np.where(y_train == c)[0]
        n_c_val = max(1, int(n_val * len(class_idx) / n_total))
        n_c_val = min(n_c_val, len(class_idx))
        selected = rng.choice(class_idx, size=n_c_val, replace=False)
        val_idx.append(selected)
    val_idx = np.concatenate(val_idx)
    val_mask = np.zeros(n_total, dtype=bool)
    val_mask[val_idx] = True
    train_mask = ~val_mask

    X_tr_raw = X_train[train_mask]
    y_tr = y_train[train_mask]
    X_val_raw = X_train[val_mask]
    y_val = y_train[val_mask]

    # Feature engineering
    X_tr = _augment(X_tr_raw)
    X_val = _augment(X_val_raw)

    mu = X_tr.mean(axis=0, keepdims=True)
    sd = X_tr.std(axis=0, keepdims=True) + 1e-6
    X_tr = (X_tr - mu) / sd
    X_val = (X_val - mu) / sd

    X_tr_t = torch.tensor(X_tr, dtype=torch.float32, device=device)
    y_tr_t = torch.tensor(y_tr, dtype=torch.long, device=device)
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.long, device=device)

    input_dim = X_tr_t.shape[1]
    hidden_dim = 768

    model = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim),
        nn.GELU(),
        ResBlock(hidden_dim, dropout=0.1),
        ResBlock(hidden_dim, dropout=0.1),
        ResBlock(hidden_dim, dropout=0.1),
        ResBlock(hidden_dim, dropout=0.05),
        nn.Linear(hidden_dim, n_classes),
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    warmup_steps = 50
    total_steps_estimate = 12000

    def lr_lambda(step):
        if step < warmup_steps:
            return max(0.01, step / warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps_estimate - warmup_steps)
        progress = min(1.0, progress)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    batch_size = 1024
    n_samples = len(X_tr_t)

    epoch = 0
    global_step = 0
    best_val_acc = 0.0
    best_state = None
    patience = 3
    epochs_without_improvement = 0

    def evaluate():
        model.eval()
        with torch.no_grad():
            preds = []
            for i in range(0, len(X_val_t), 16384):
                xb = X_val_t[i:i+16384]
                preds.append(model(xb).argmax(dim=1))
            preds = torch.cat(preds)
            return (preds == y_val_t).float().mean().item()

    while time.time() - start < time_budget_s * 0.85:
        model.train()
        perm = torch.randperm(n_samples, device=device)
        total = 0.0
        for i in range(0, n_samples, batch_size):
            idx = perm[i:i + batch_size]
            xb = X_tr_t[idx]
            yb = y_tr_t[idx]
            out = model(xb)
            loss = loss_fn(out, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            scheduler.step()
            global_step += 1
            total += loss.item() * len(idx)
            if time.time() - start > time_budget_s * 0.85:
                break
        epoch += 1

        val_acc = evaluate()
        model.train()
        log(epoch, train_loss=total / n_samples, holdout_acc=val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping at epoch {epoch}, best val acc: {best_val_acc:.5f}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Prediction function
    mu_t = torch.tensor(mu, dtype=torch.float32, device=device)
    sd_t = torch.tensor(sd, dtype=torch.float32, device=device)

    def predict(X_new):
        model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(X_new), 32768):
                xb = torch.tensor(X_new[i:i + 32768], dtype=torch.float32, device=device)
                xb_aug = _augment_np(xb)
                xb_aug = (xb_aug - mu_t) / sd_t
                out.append(model(xb_aug).argmax(dim=1).cpu().numpy())
        return np.concatenate(out)

    return predict


def _augment_np(X):
    """Apply feature engineering to a torch tensor (for prediction)."""
    elevation = X[:, 0:1]
    slope = X[:, 2:3]
    wild_interactions = elevation * X[:, 10:14]
    top_soil_indices = torch.tensor([42, 36, 45, 46, 23, 17, 15, 16, 30, 19, 43, 26, 51, 52, 53, 25, 35, 49], device=X.device)
    soil_interactions = elevation * X[:, top_soil_indices]
    slope_soil = slope * X[:, torch.tensor([23, 17, 15, 19, 16, 30], device=X.device)]
    elev_sq = elevation ** 2
    elev_cu = elevation ** 3
    return torch.cat([X, wild_interactions, soil_interactions, slope_soil, elev_sq, elev_cu], dim=1)
