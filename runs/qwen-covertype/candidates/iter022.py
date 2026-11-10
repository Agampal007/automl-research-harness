"""Neural network approach for covertype.

Hypothesis: A regularized MLP can capture smooth interactions between continuous
features (elevation, hillshade) and binary soil/wilderness indicators more
efficiently than axis-aligned tree splits, especially for rare classes.
"""

import time
import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def _standardize(X, train_mean, train_std):
    return (X - train_mean) / (train_std + 1e-8)


def fit(X_train, y_train, *, seed, device, time_budget_s, log):
    start = time.time()
    torch.manual_seed(seed)
    np.random.seed(seed)

    X = np.ascontiguousarray(X_train, dtype=np.float32)
    y = np.ascontiguousarray(y_train, dtype=np.int64)
    n = len(X)
    n_features = X.shape[1]
    n_classes = int(y.max()) + 1

    # Standardize continuous features (first 10), leave binary as-is
    cont_mean = X[:, :10].mean(axis=0).astype(np.float32)
    cont_std = X[:, :10].std(axis=0).astype(np.float32)
    X_std = X.copy()
    X_std[:, :10] = (X[:, :10] - cont_mean) / (cont_std + 1e-8)
    X_std = np.ascontiguousarray(X_std, dtype=np.float32)

    # Split off a holdout for early stopping / monitoring
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    n_val = min(20000, n // 10)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]

    X_tr = torch.from_numpy(X_std[tr_idx]).to(device)
    y_tr = torch.from_numpy(y[tr_idx]).to(device)
    X_val = torch.from_numpy(X_std[val_idx]).to(device)
    y_val = torch.from_numpy(y[val_idx]).to(device)

    # Model: 4 hidden layers with batch norm and dropout
    model = nn.Sequential(
        nn.Linear(n_features, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(256, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Linear(128, n_classes),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss()

    # DataLoader
    dataset = TensorDataset(X_tr, y_tr)
    batch_size = 4096
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    # Time budget for training
    train_budget = time_budget_s * 0.80
    best_val_acc = 0.0
    best_state = None
    patience = 5
    bad_epochs = 0
    epoch = 0

    while time.time() - start < train_budget:
        epoch += 1
        model.train()
        total_loss = 0.0
        n_batches = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        scheduler.step()

        # Evaluate on holdout
        model.eval()
        with torch.no_grad():
            val_out = model(X_val)
            val_pred = val_out.argmax(dim=1)
            val_acc = float((val_pred == y_val).float().mean().item())

        avg_loss = total_loss / max(n_batches, 1)
        log(epoch, train_loss=avg_loss, holdout_acc=val_acc)
        print(f"epoch {epoch}: loss={avg_loss:.4f}, val_acc={val_acc:.5f}, t={time.time()-start:.1f}s")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    print(f"Best val acc: {best_val_acc:.5f}, epochs={epoch}")

    # Final: retrain on ALL data for a few more epochs with the best LR
    # Actually, let's just use the best model we found
    # But we should retrain on all data to use all the information
    # Let's do a quick fine-tune on all data
    X_all = torch.from_numpy(X_std).to(device)
    y_all = torch.from_numpy(y).to(device)
    dataset_all = TensorDataset(X_all, y_all)
    loader_all = DataLoader(dataset_all, batch_size=batch_size, shuffle=True)

    # Fine-tune for a few epochs on all data
    fine_budget = min(time_budget_s * 0.15, 15.0)
    fine_start = time.time()
    fine_epochs = 0
    while time.time() - fine_start < fine_budget and fine_epochs < 3:
        fine_epochs += 1
        model.train()
        for xb, yb in loader_all:
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
    model.eval()
    print(f"Fine-tuned {fine_epochs} epochs on all data")

    def predict(X_new):
        X_new = np.ascontiguousarray(X_new, dtype=np.float32)
        # Standardize
        Xs = X_new.copy()
        Xs[:, :10] = (Xs[:, :10] - cont_mean) / (cont_std + 1e-8)
        Xs = np.ascontiguousarray(Xs, dtype=np.float32)
        Xt = torch.from_numpy(Xs).to(device)
        model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(Xt), 65536):
                out.append(model(Xt[i:i+65536]).argmax(dim=1).cpu().numpy())
        return np.concatenate(out)

    return predict
