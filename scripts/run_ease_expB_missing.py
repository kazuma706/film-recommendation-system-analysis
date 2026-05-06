"""
Run EASE Experiment B for sparsity 0.5 and 1.0 (previously skipped), update ease_expB_results.csv.
"""

import gc
import random
import warnings
import os
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.linalg as linalg
from sklearn.model_selection import train_test_split

DATA_DIR   = '/Users/kazuma/Desktop/film data analysis/ml-25m/'
OUTPUT_DIR = '/Users/kazuma/Desktop/film data analysis/popularity-bias-project/outputs/'

RANDOM_SEED  = 42
N_EVAL_USERS = 500
N_BOOTSTRAP  = 1_000
TOP_N        = 10
LAMBDA_EASE  = 500.0
MIN_RATINGS  = 50


# EASE class (identical to notebook)

class EASE:
    def __init__(self, lambda_=500.0, min_ratings=50):
        self.lambda_ = lambda_
        self.min_ratings = min_ratings
        self.B = None
        self.item_ids = None
        self.item_to_idx = {}

    def fit(self, X, item_ids):
        n_ratings_per_item = np.asarray((X > 0).sum(axis=0)).ravel()
        mask   = n_ratings_per_item >= self.min_ratings
        n_kept = int(mask.sum())
        if n_kept == 0:
            warnings.warn(f'EASE.fit: no items passed min_ratings={self.min_ratings}.')
            return self
        X_f = X[:, mask]
        self.item_ids    = item_ids[mask].copy()
        self.item_to_idx = {int(iid): i for i, iid in enumerate(self.item_ids)}
        G = np.asarray((X_f.T @ X_f).todense(), dtype=np.float32)
        del X_f; gc.collect()
        G[np.diag_indices(n_kept)] += np.float32(self.lambda_)
        P = linalg.inv(G)
        del G; gc.collect()
        diag_P = np.diag(P).copy()
        B = -P / diag_P[np.newaxis, :]
        np.fill_diagonal(B, 0.0)
        del P; gc.collect()
        self.B = B.astype(np.float32)
        del B; gc.collect()
        return self

    def recommend(self, user_train_ratings, seen_ids, n=10):
        if self.B is None or self.item_ids is None:
            return []
        r = np.zeros(len(self.item_ids), dtype=np.float32)
        for iid, rating in user_train_ratings.items():
            idx = self.item_to_idx.get(int(iid))
            if idx is not None:
                r[idx] = float(rating) / 5.0
        scores = r @ self.B
        ranked = sorted(
            ((int(self.item_ids[i]), float(scores[i]))
             for i in range(len(self.item_ids))
             if int(self.item_ids[i]) not in seen_ids),
            key=lambda x: x[1], reverse=True,
        )
        return [mid for mid, _ in ranked[:n]]


# helper functions

def ndcg_at_k(recommended, relevant_ratings, k=10):
    if not relevant_ratings:
        return 0.0
    dcg = sum(
        (relevant_ratings[item] / 5.0) / np.log2(rank + 2)
        for rank, item in enumerate(recommended[:k])
        if item in relevant_ratings
    )
    ideal_rels = sorted(relevant_ratings.values(), reverse=True)[:k]
    idcg = sum(
        (rel / 5.0) / np.log2(rank + 2)
        for rank, rel in enumerate(ideal_rels)
    )
    return dcg / idcg if idcg > 0 else 0.0


def bootstrap_ci(values, n_boot=N_BOOTSTRAP, ci=0.95):
    if len(values) == 0:
        return (np.nan, np.nan)
    rng  = np.random.default_rng(RANDOM_SEED)
    arr  = np.asarray(values, dtype=float)
    boot = [np.mean(rng.choice(arr, size=len(arr), replace=True))
            for _ in range(n_boot)]
    alpha = (1 - ci) / 2
    return (float(np.percentile(boot, alpha * 100)),
            float(np.percentile(boot, (1 - alpha) * 100)))


def build_sparse_matrix(df):
    all_item_ids = sorted(df['movieId'].unique())
    all_user_ids = sorted(df['userId'].unique())
    item_to_col  = {int(iid): i for i, iid in enumerate(all_item_ids)}
    user_to_row  = {int(uid): i for i, uid in enumerate(all_user_ids)}
    rows = df['userId'].map(user_to_row).values.astype(np.int32)
    cols = df['movieId'].map(item_to_col).values.astype(np.int32)
    vals = (df['rating'].values / 5.0).astype(np.float32)
    X = sp.csr_matrix(
        (vals, (rows, cols)),
        shape=(len(all_user_ids), len(all_item_ids)),
        dtype=np.float32,
    )
    item_ids = np.array(all_item_ids, dtype=np.int64)
    return X, item_ids, user_to_row


def memory_estimate(X, min_ratings=MIN_RATINGS):
    n_per_item = np.asarray((X > 0).sum(axis=0)).ravel()
    n_kept     = int((n_per_item >= min_ratings).sum())
    gram_gb    = n_kept ** 2 * 4 / 1e9
    peak_gb    = gram_gb * 2
    print(f'  [MEM] Items passing min_ratings={min_ratings} filter : {n_kept:,}')
    print(f'  [MEM] Gram matrix (float32, {n_kept}×{n_kept})       : {gram_gb:.2f} GB')
    print(f'  [MEM] Peak estimate (G + P simultaneously)           : {peak_gb:.2f} GB')
    return n_kept


def fixed_catalogue_sample(df, sparsity, random_seed=RANDOM_SEED):
    return (
        df.groupby('movieId', group_keys=False)
        .apply(lambda x: x.sample(
            n=max(1, int(len(x) * sparsity)),
            random_state=random_seed,
        ))
        .reset_index(drop=True)
    )


# main

def run_expB_level(full_ratings, sparsity):
    print(f'\n{"="*62}')
    print(f'EXP B  sparsity={sparsity}')
    print(f'{"="*62}')

    subset_B = fixed_catalogue_sample(full_ratings, sparsity)
    avg_rpm  = len(subset_B) / subset_B['movieId'].nunique()
    print(f'  n_ratings           : {len(subset_B):,}')
    print(f'  Unique movies       : {subset_B["movieId"].nunique():,}')
    print(f'  Avg ratings / movie : {avg_rpm:.2f}')

    train_df, test_df = train_test_split(
        subset_B, test_size=0.2, random_state=RANDOM_SEED
    )

    X_B, item_ids_B, _ = build_sparse_matrix(train_df)
    print(f'  Train matrix : {X_B.shape}  nnz={X_B.nnz:,}  dtype={X_B.dtype}')

    n_kept_est = memory_estimate(X_B, MIN_RATINGS)

    ease_B = EASE(lambda_=LAMBDA_EASE, min_ratings=MIN_RATINGS)
    try:
        ease_B.fit(X_B, item_ids_B)
    except MemoryError:
        msg = (f'MemoryError at sparsity={sparsity} '
               f'(est. {n_kept_est:,} items, Gram ~{n_kept_est**2*4/1e9:.1f} GB).')
        print(f'  !! {msg}')
        del X_B; gc.collect()
        return dict(sparsity=sparsity, n_ratings=len(subset_B),
                    ndcg=np.nan, ci_lower=np.nan, ci_upper=np.nan)
    finally:
        del X_B; gc.collect()

    if ease_B.B is None:
        print('  No items passed filter — recording NaN.')
        return dict(sparsity=sparsity, n_ratings=len(subset_B),
                    ndcg=np.nan, ci_lower=np.nan, ci_upper=np.nan)

    print(f'  EASE B matrix : {len(ease_B.item_ids):,} items  '
          f'({ease_B.B.nbytes / 1e6:.0f} MB at float32)')

    eval_users = list(set(train_df['userId'].unique()) &
                      set(test_df['userId'].unique()))
    random.seed(RANDOM_SEED)
    eval_sample = random.sample(eval_users, min(N_EVAL_USERS, len(eval_users)))
    print(f'  Evaluation users : {len(eval_sample)}')

    test_lookup = (
        test_df.groupby('userId', group_keys=False)
        .apply(lambda x: dict(zip(x['movieId'].astype(int), x['rating'])))
        .to_dict()
    )
    train_ratings_lookup = (
        train_df.groupby('userId', group_keys=False)
        .apply(lambda x: dict(zip(x['movieId'].astype(int), x['rating'])))
        .to_dict()
    )
    train_seen_lookup = (
        train_df.groupby('userId')['movieId'].apply(set).to_dict()
    )

    ndcg_vals = []
    for i, uid in enumerate(eval_sample):
        if i % 100 == 0:
            print(f'  ... user {i}/{len(eval_sample)}', flush=True)
        relevant = test_lookup.get(uid, {})
        if not relevant:
            continue
        recs = ease_B.recommend(
            train_ratings_lookup.get(uid, {}),
            train_seen_lookup.get(uid, set()),
            n=TOP_N,
        )
        ndcg_vals.append(ndcg_at_k(recs, relevant))

    mean_ndcg    = float(np.mean(ndcg_vals)) if ndcg_vals else np.nan
    ci_lo, ci_hi = bootstrap_ci(ndcg_vals)
    print(f'  NDCG@10 = {mean_ndcg:.4f}  CI=[{ci_lo:.4f}, {ci_hi:.4f}]  '
          f'(n={len(ndcg_vals)} users)')

    del ease_B; gc.collect()
    return dict(
        sparsity=sparsity,
        n_ratings=len(subset_B),
        ndcg=round(mean_ndcg, 4),
        ci_lower=round(ci_lo, 4),
        ci_upper=round(ci_hi, 4),
    )


if __name__ == '__main__':
    print('Loading full 25M ratings...')
    full_ratings = pd.read_csv(DATA_DIR + 'ratings.csv')
    print(f'  {len(full_ratings):,} ratings loaded')

    # load existing results and patch the two NaN rows
    csv_path = OUTPUT_DIR + 'ease_expB_results.csv'
    df_existing = pd.read_csv(csv_path)
    print(f'\nExisting ease_expB_results.csv:\n{df_existing.to_string(index=False)}')

    new_rows = []
    for sparsity in [0.5, 1.0]:
        result = run_expB_level(full_ratings, sparsity)
        new_rows.append(result)

    # replace the NaN rows in the existing CSV
    df_new = pd.DataFrame(new_rows)
    df_updated = df_existing[~df_existing['sparsity'].isin([0.5, 1.0])].copy()
    df_updated = pd.concat([df_updated, df_new], ignore_index=True)
    df_updated = df_updated.sort_values('sparsity').reset_index(drop=True)

    df_updated.to_csv(csv_path, index=False)
    print(f'\nUpdated {csv_path}:')
    print(df_updated.to_string(index=False))

    # regenerate HTML visualizations
    print('\nRegenerating HTML visualizations...')
    import subprocess
    result = subprocess.run(
        ['python', 'scripts/update_viz_ease.py'],
        capture_output=True, text=True,
        cwd='/Users/kazuma/Desktop/film data analysis/popularity-bias-project',
    )
    print(result.stdout)
    if result.stderr:
        print('STDERR:', result.stderr[:500])

    print('\nAll done.')
