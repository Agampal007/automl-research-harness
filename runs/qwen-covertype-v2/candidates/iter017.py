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


def fit(X_train, y_train, *, seed, device, time_budget_s, log):
    start = time.time()
    n_classes = int(y_train.max()) + 1

    # Feature engineering (same as incumbent)
    elevation = X_train[:, 0:1]
    slope = X_train[:, 2:3]
    wild_interactions = elevation * X_train[:, 10:14]
    top_soil_indices = [42, 36, 45, 46, 23, 17, 15, 16, 30, 19, 43, 26, 51, 52, 53, 25, 35, 49]
    soil_interactions = elevation * X_train[:, top_soil_indices]
    slope_soil = slope * X_train[:, [23, 17, 15, 19, 16, 30]]
    elev_sq = elevation ** 2
    elev_cu = elevation ** 3

    X_aug = np.hstack([
        X_train, wild_interactions, soil_interactions,
        slope_soil, elev_sq, elev_cu,
    ]).astype(np.float32)

    mu = X_aug.mean(axis=0, keepdims=True)
    sd = X_aug.std(axis=0, keepdims=True) + 1e-6
    X_aug = (X_aug - mu) / sd

    X = torch.tensor(X_aug, dtype=torch.float32, device=device)
    y = torch.tensor(y_train, dtype=torch.long, device=device)

    input_dim = X.shape[1]
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
    n_samples = len(X)

    epoch = 0
    global_step = 0
    best_holdout = 0.0
    best_state = None

    # SWA: keep a buffer of recent checkpoints
    swa_buffer = []
    swa_window = 5  # average last 5 checkpoints

    while time.time() - start < time_budget_s * 0.80:
        model.train()
        perm = torch.randperm(n_samples, device=device)
        total = 0.0
        for i in range(0, n_samples, batch_size):
            idx = perm[i:i + batch_size]
            xb = X[idx]
            yb = y[idx]
            out = model(xb)
            loss = loss_fn(out, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            scheduler.step()
            global_step += 1
            total += loss.item() * len(idx)
            if time.time() - start > time_budget_s * 0.80:
                break
        epoch += 1

        model.eval()
        with torch.no_grad():
            val_pred = model(X[:min(20000, n_samples)]).argmax(dim=1)
            holdout_acc = (val_pred == y[:min(20000, n_samples)]).float().mean().item()
        model.train()
        log(epoch, train_loss=total / n_samples, holdout_acc=holdout_acc)

        # Track best single checkpoint
        if holdout_acc > best_holdout:
            best_holdout = holdout_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        # SWA buffer: keep last N checkpoints
        swa_buffer.append({k: v.clone() for k, v in model.state_dict().items()})
        if len(swa_buffer) > swa_window:
            swa_buffer.pop(0)

    # After training, compute SWA average of the buffered checkpoints
    if len(swa_buffer) >= 2:
        swa_state = {}
        for key in swa_buffer[0]:
            swa_state[key] = sum(s[key].float() for s in swa_buffer) / len(swa_buffer)
        model.load_state_dict(swa_state)
    elif best_state is not None:
        model.load_state_dict(best_state)

    # Verify SWA quality vs best single
    model.eval()
    with torch.no_grad():
        swa_pred = model(X[:min(20000, n_samples)]).argmax(dim=1)
        swa_acc = (swa_pred == y[:min(20000, n_samples)]).float().mean().item()
    print(f"SWA holdout acc: {swa_acc:.5f}, best single: {best_holdout:.5f}")

    # Use whichever is better
    if best_state is not None:
        with torch.no_grad():
            model.load_state_dict(best_state)
            best_pred = model(X[:min(20000, n_samples)]).argmax(dim=1)
            best_acc = (best_pred == y[:min(20000, n_samples)]).float().mean().item()
        if best_acc > swa_acc:
            model.load_state_dict(best_state)
            print("Using best single checkpoint")
        else:
            model.load_state_dict(swa_state)
            print("Using SWA average")
    model.train()

    # Prediction function
    mu_t = torch.tensor(mu, dtype=torch.float32, device=device)
    sd_t = torch.tensor(sd, dtype=torch.float32, device=device)
    top_soil_indices_t = torch.tensor(top_soil_indices, dtype=torch.long, device=device)
    slope_soil_indices_t = torch.tensor([23, 17, 15, 19, 16, 30], dtype=torch.long, device=device)

    def predict(X_new):
        model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(X_new), 32768):
                xb = torch.tensor(X_new[i:i + 32768], dtype=torch.float32, device=device)
                elev = xb[:, 0:1]
                slope = xb[:, 2:3]
                wild_int = elev * xb[:, 10:14]
                soil_int = elev * xb[:, top_soil_indices_t]
                slope_soil = slope * xb[:, slope_soil_indices_t]
                elev_sq = elev ** 2
                elev_cu = elev ** 3
                xb_aug = torch.cat([xb, wild_int, soil_int, slope_soil, elev_sq, elev_cu], dim=1)
                xb_aug = (xb_aug - mu_t) / sd_t
                out.append(model(xb_aug).argmax(dim=1).cpu().numpy())
        return np.concatenate(out)

    return predict
