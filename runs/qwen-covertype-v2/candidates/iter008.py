import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingClassifier


def fit(X_train, y_train, *, seed, device, time_budget_s, log):
    start = time.time()
    n_classes = int(y_train.max()) + 1

    # Feature engineering
    elevation = X_train[:, 0:1]
    slope = X_train[:, 2:3]
    wild_interactions = elevation * X_train[:, 10:14]
    top_soil_indices = [42, 36, 45, 46, 23, 17, 15, 16, 30, 19, 43, 26, 51, 52, 53, 25, 35, 49]
    soil_interactions = elevation * X_train[:, top_soil_indices]
    slope_soil = slope * X_train[:, [23, 17, 15, 19, 16, 30]]
    elev_sq = elevation ** 2
    elev_cu = elevation ** 3

    X_aug = np.hstack([
        X_train, wild_interactions, soil_interactions, slope_soil, elev_sq, elev_cu
    ]).astype(np.float32)

    mu = X_aug.mean(axis=0, keepdims=True)
    sd = X_aug.std(axis=0, keepdims=True) + 1e-6
    X_aug_norm = (X_aug - mu) / sd

    # === MLP training (65% of time budget) ===
    mlp_deadline = start + time_budget_s * 0.65

    X = torch.tensor(X_aug_norm, dtype=torch.float32, device=device)
    y = torch.tensor(y_train, dtype=torch.long, device=device)

    input_dim = X.shape[1]
    model = nn.Sequential(
        nn.Linear(input_dim, 768), nn.BatchNorm1d(768), nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(768, 768), nn.BatchNorm1d(768), nn.ReLU(), nn.Dropout(0.1),
        nn.Linear(768, 384), nn.BatchNorm1d(384), nn.ReLU(), nn.Dropout(0.05),
        nn.Linear(384, n_classes),
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=5e-3, weight_decay=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    warmup_steps = 50
    total_steps_estimate = 3000
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps_estimate - warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    batch_size = 1024
    n_samples = len(X)
    epoch = 0
    last_loss = 0.0

    while time.time() < mlp_deadline:
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
            total += loss.item() * len(idx)
            if time.time() > mlp_deadline:
                break
        epoch += 1
        last_loss = total / n_samples

        if epoch % 3 == 0:
            model.eval()
            with torch.no_grad():
                val_pred = model(X[:20000]).argmax(dim=1)
                holdout_acc = (val_pred == y[:20000]).float().mean().item()
            model.train()
            log(epoch, train_loss=last_loss, holdout_acc=holdout_acc)

    # === HistGBM training (30% of time budget) ===
    gbm_deadline = start + time_budget_s * 0.95
    gbm_time_available = max(5, gbm_deadline - time.time())

    # Estimate iterations based on available time
    # From iteration 7: ~500 iterations in 80s => ~6.25 iter/s
    est_iters_per_sec = 6.0
    max_iters = int(gbm_time_available * est_iters_per_sec * 0.8)
    max_iters = max(50, min(max_iters, 300))

    hgb = HistGradientBoostingClassifier(
        max_iter=max_iters,
        learning_rate=0.1,
        max_depth=6,
        min_samples_leaf=50,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        random_state=seed,
    )
    hgb.fit(X_aug, y_train)

    log(epoch + 100, train_loss=last_loss, holdout_acc=0.0)
    print(f"MLP epochs: {epoch}, HistGBM iters: {max_iters}, time elapsed: {time.time()-start:.1f}s")

    # === Predict function ===
    model.eval()
    mu_t = torch.tensor(mu, dtype=torch.float32, device=device)
    sd_t = torch.tensor(sd, dtype=torch.float32, device=device)
    top_soil_indices_t = torch.tensor(top_soil_indices, dtype=torch.long, device=device)
    slope_soil_indices_t = torch.tensor([23, 17, 15, 19, 16, 30], dtype=torch.long, device=device)

    def predict(X_new):
        X_new = np.asarray(X_new, dtype=np.float32)
        # MLP predictions (softmax)
        mlp_probs = []
        with torch.no_grad():
            for i in range(0, len(X_new), 32768):
                xb = torch.tensor(X_new[i:i+32768], dtype=torch.float32, device=device)
                elev = xb[:, 0:1]
                slope = xb[:, 2:3]
                wild_int = elev * xb[:, 10:14]
                soil_int = elev * xb[:, top_soil_indices_t]
                slope_soil = slope * xb[:, slope_soil_indices_t]
                elev_sq = elev ** 2
                elev_cu = elev ** 3
                xb_aug = torch.cat([xb, wild_int, soil_int, slope_soil, elev_sq, elev_cu], dim=1)
                xb_aug = (xb_aug - mu_t) / sd_t
                mlp_probs.append(torch.softmax(model(xb_aug), dim=1).cpu().numpy())
        mlp_probs = np.concatenate(mlp_probs)

        # HistGBM predictions
        elevation = X_new[:, 0:1]
        slope = X_new[:, 2:3]
        wild_int = elevation * X_new[:, 10:14]
        soil_int = elevation * X_new[:, top_soil_indices]
        slope_soil = slope * X_new[:, [23, 17, 15, 19, 16, 30]]
        elev_sq = elevation ** 2
        elev_cu = elevation ** 3
        X_aug_new = np.hstack([X_new, wild_int, soil_int, slope_soil, elev_sq, elev_cu]).astype(np.float32)
        gbm_probs = hgb.predict_proba(X_aug_new)

        # Weighted ensemble
        ensemble_probs = 0.6 * mlp_probs + 0.4 * gbm_probs
        return ensemble_probs.argmax(axis=1)

    return predict
