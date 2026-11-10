"""Seed solution: a plain MLP. Deliberately unremarkable, so the researcher has room to work."""

import time

import numpy as np
import torch
import torch.nn as nn


def fit(X_train, y_train, *, seed, device, time_budget_s, log):
    start = time.time()
    n_classes = int(y_train.max()) + 1
    mu = X_train.mean(axis=0, keepdims=True)
    sd = X_train.std(axis=0, keepdims=True) + 1e-6

    X = torch.tensor((X_train - mu) / sd, dtype=torch.float32, device=device)
    y = torch.tensor(y_train, dtype=torch.long, device=device)

    model = nn.Sequential(
        nn.Linear(X.shape[1], 256), nn.ReLU(),
        nn.Linear(256, 256), nn.ReLU(),
        nn.Linear(256, n_classes),
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    batch, epoch = 1024, 0
    while time.time() - start < time_budget_s * 0.85:
        perm = torch.randperm(len(X), device=device)
        total = 0.0
        for i in range(0, len(X), batch):
            idx = perm[i:i + batch]
            loss = loss_fn(model(X[idx]), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
            if time.time() - start > time_budget_s * 0.85:
                break
        epoch += 1
        log(epoch, train_loss=total / len(X))

    mu_t = torch.tensor(mu, dtype=torch.float32, device=device)
    sd_t = torch.tensor(sd, dtype=torch.float32, device=device)

    def predict(X_new):
        model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(X_new), 65536):
                xb = torch.tensor(X_new[i:i + 65536], dtype=torch.float32, device=device)
                out.append(model((xb - mu_t) / sd_t).argmax(dim=1).cpu().numpy())
        return np.concatenate(out)

    return predict
