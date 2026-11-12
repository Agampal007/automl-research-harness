"""Probe: understand class separability and feature importance structure."""
import numpy as np
import time

def probe(X_train, y_train):
    n, d = X_train.shape
    classes = np.unique(y_train)
    print(f"Shape: {n} x {d}, classes: {classes}")
    print(f"Class counts: {[int((y_train==c).sum()) for c in classes]}")
    
    # Per-class means for continuous features (first 10)
    cont_names = ["Elev","Aspect","Slope","H_Dist_Hyd","V_Dist_Hyd","H_Dist_Road","Hill9","HillNoon","Hill3","H_Dist_Fire"]
    print("\n=== Per-class means (continuous features) ===")
    print(f"{'Class':<6}", end="")
    for name in cont_names:
        print(f"{name:>12}", end="")
    print()
    for c in classes:
        mask = (y_train == c)
        means = X_train[mask, :10].mean(axis=0)
        print(f"{c:<6}", end="")
        for m in means:
            print(f"{m:>12.1f}", end="")
        print()
    
    # Per-class std for continuous features
    print("\n=== Per-class std (continuous features) ===")
    print(f"{'Class':<6}", end="")
    for name in cont_names:
        print(f"{name:>12}", end="")
    print()
    for c in classes:
        mask = (y_train == c)
        stds = X_train[mask, :10].std(axis=0)
        print(f"{c:<6}", end="")
        for s in stds:
            print(f"{s:>12.1f}", end="")
        print()
    
    # Which soil types are most associated with each class?
    print("\n=== Top 3 soil types per class ===")
    soil_cols = np.arange(14, 54)  # soil type columns
    for c in classes:
        mask = (y_train == c)
        soil_means = X_train[mask][:, soil_cols].mean(axis=0)
        top3 = np.argsort(soil_means)[::-1][:3]
        top3_soil_ids = top3 + 14
        top3_vals = soil_means[top3]
        print(f"Class {c}: soils {top3_soil_ids} with freqs {top3_vals}")
    
    # Which wilderness areas per class?
    print("\n=== Wilderness area distribution per class ===")
    wild_cols = np.arange(10, 14)
    for c in classes:
        mask = (y_train == c)
        wild_means = X_train[mask][:, wild_cols].mean(axis=0)
        print(f"Class {c}: {wild_means}")
    
    # Check: how well does elevation alone separate?
    print("\n=== Elevation percentiles per class ===")
    for c in classes:
        mask = (y_train == c)
        elevs = X_train[mask, 0]
        print(f"Class {c}: min={elevs.min():.0f}, p25={np.percentile(elevs,25):.0f}, "
              f"median={np.median(elevs):.0f}, p75={np.percentile(elevs,75):.0f}, max={elevs.max():.0f}")
    
    # Simple: train a quick logistic regression to see feature importances
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    print("\n=== Logistic Regression (quick) ===")
    t0 = time.time()
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train)
    lr = LogisticRegression(max_iter=200, C=1.0, solver='lbfgs')
    lr.fit(Xs, y_train)
    print(f"LR train acc: {lr.score(Xs, y_train):.4f} (took {time.time()-t0:.1f}s)")
    
    # Feature importances from LR (use absolute weight * feature_mean)
    coefs = lr.coef_  # (7, 54)
    # For each feature, max absolute coef across classes
    feat_importance = np.abs(coefs).max(axis=0)
    top10_idx = np.argsort(feat_importance)[::-1][:15]
    print(f"Top 15 features by max |coef|: {top10_idx}")
    print(f"Their importances: {feat_importance[top10_idx]}")
    
    # Check overlap: for each pair of classes, what's the mean distance in feature space?
    print("\n=== Pairwise mean distances (standardized) ===")
    class_means_list = []
    for c in classes:
        class_means_list.append(Xs[y_train==c].mean(axis=0))
    class_means = np.array(class_means_list)
    dists = np.sqrt(((class_means[:, None, :] - class_means[None, :, :])**2).sum(axis=2))
    print(f"{'':>6}", end="")
    for c in classes:
        print(f"{c:>6}", end="")
    print()
    for i, ci in enumerate(classes):
        print(f"{ci:>6}", end="")
        for j, cj in enumerate(classes):
            print(f"{dists[i,j]:>6.1f}", end="")
        print()
    
    # Check: are there any features that are constant or near-constant?
    print("\n=== Feature variance ===")
    variances = X_train.var(axis=0)
    low_var = np.where(variances < 0.01)[0]
    print(f"Features with variance < 0.01: {low_var}")
    
    # Check: how many rows have exactly the same feature values? (duplicates)
    print("\n=== Duplicate check ===")
    # Sample-based check
    sample_idx = np.random.choice(n, min(50000, n), replace=False)
    sample = X_train[sample_idx]
    from collections import Counter
    # Use a hash approach
    rounded = (sample * 100).astype(np.int64)
    hashes = [tuple(row) for row in rounded[:10000]]
    dup_count = len(hashes) - len(set(hashes))
    print(f"Among first 10000 of 50000 sampled rows: {dup_count} duplicates")
    
    print("\nProbe complete.")
