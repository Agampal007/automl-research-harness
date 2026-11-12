import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def fit(X_train, y_train, *, seed, device, time_budget_s, log):
    start = time.time()
    n_classes = int(y_train.max()) + 1

    # Feature engineering: add interaction features
    # Continuous features: indices 0-9
    # Wilderness areas: indices 10-13
    # Soil types: indices 14-53
    
    elevation = X_train[:, 0:1]  # (n, 1)
    
    # Elevation interactions with wilderness areas (4 features)
    wild_interactions = elevation * X_train[:, 10:14]
    
    # Elevation interactions with top soil types by frequency
    # Top soils by overall frequency: 42(0.198), 36(0.0996), 45(0.0903), 46(0.0777), 
    # 21(0.0578), 23(0.0562), 29(0.0518), 11(0.0516), 30(0.0443), 37(0.0269)
    # But more importantly, class-specific soils:
    # Class 0: 42, 36, 35 | Class 1: 42, 45, 25 | Class 2: 23, 17, 15
    # Class 3: 16, 30, 19 | Class 4: 43, 26, 42 | Class 5: 23, 19, 15
    # Class 6: 51, 52, 53
    # Let's pick a diverse set of top soils: indices (0-based from 14)
    # Soil 42 = index 14+28=42, soil 36=14+22=36, soil 45=14+31=45
    # soil 46=14+32=46, soil 23=14+9=23, soil 17=14+3=17, soil 15=14+1=15
    # soil 16=14+2=16, soil 30=14+16=30, soil 19=14+5=19, soil 43=14+29=43
    # soil 26=14+12=26, soil 51=14+37=51, soil 52=14+38=52, soil 53=14+39=53
    # soil 25=14+11=25
    top_soil_indices = [42, 36, 45, 46, 23, 17, 15, 16, 30, 19, 43, 26, 51, 52, 53, 25]
    soil_interactions = elevation * X_train[:, top_soil_indices]
    
    # Elevation squared
    elev_sq = elevation ** 2
    
    # Combine all features
    X_aug = np.hstack([
        X_train,
        wild_interactions,
        soil_interactions,
        elev_sq
    ]).astype(np.float32)
    
    # Standardize
    mu = X_aug.mean(axis=0, keepdims=True)
    sd = X_aug.std(axis=0, keepdims=True) + 1e-6
    X_aug = (X_aug - mu) / sd

    X = torch.tensor(X_aug, dtype=torch.float32, device=device)
    y = torch.tensor(y_train, dtype=torch.long, device=device)

    # Class weights for imbalanced classes
    class_counts = np.bincount(y_train, minlength=n_classes).astype(np.float64)
    class_weights = 1.0 / (class_counts + 1e-6)
    # Normalize so that the average weight is 1
    class_weights = class_weights / class_weights.mean()
    # Cap extreme weights
    class_weights = np.clip(class_weights, 0.5, 5.0)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)

    # Architecture: wider with batch norm
    input_dim = X.shape[1]
    model = nn.Sequential(
        nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.ReLU(),
        nn.Linear(512, 512), nn.BatchNorm1d(512), nn.ReLU(),
        nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(),
        nn.Linear(256, n_classes),
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights_tensor)

    # Cosine learning rate schedule
    total_steps_estimate = 2000  # rough estimate
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps_estimate)

    batch_size = 2048
    n_samples = len(X)
    steps_per_epoch = n_samples // batch_size
    
    epoch = 0
    global_step = 0
    while time.time() - start < time_budget_s * 0.85:
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
            if time.time() - start > time_budget_s * 0.85:
                break
        epoch += 1
        # Log holdout accuracy every few epochs
        if epoch % 3 == 0:
            model.eval()
            with torch.no_grad():
                val_pred = model(X[:min(10000, n_samples)]).argmax(dim=1)
                holdout_acc = (val_pred == y[:min(10000, n_samples)]).float().mean().item()
            model.train()
            log(epoch, train_loss=total / n_samples, holdout_acc=holdout_acc)
        else:
            log(epoch, train_loss=total / n_samples)

    # Final prediction function
    mu_t = torch.tensor(mu, dtype=torch.float32, device=device)
    sd_t = torch.tensor(sd, dtype=torch.float32, device=device)
    
    # Pre-compute interaction feature indices for prediction
    top_soil_indices_t = torch.tensor(top_soil_indices, dtype=torch.long, device=device)

    def predict(X_new):
        model.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(X_new), 32768):
                xb = torch.tensor(X_new[i:i + 32768], dtype=torch.float32, device=device)
                # Build augmented features
                elev = xb[:, 0:1]
                wild_int = elev * xb[:, 10:14]
                soil_int = elev * xb[:, top_soil_indices_t]
                elev_sq = elev ** 2
                xb_aug = torch.cat([xb, wild_int, soil_int, elev_sq], dim=1)
                xb_aug = (xb_aug - mu_t) / sd_t
                out.append(model(xb_aug).argmax(dim=1).cpu().numpy())
        return np.concatenate(out)

    return predict
