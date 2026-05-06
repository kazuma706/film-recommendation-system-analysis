"""
Compute catalogue and long-tail coverage for EASE, update hybrid_summary.csv.
"""

import gc
import random
import warnings
import subprocess

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
N_SAMPLE     = 2_000_000   # same as notebook 02_models.ipynb


# EASE (identical to notebook)

class EASE:
    def __init__(self, lambda_=500.0, min_ratings=50):
        self.lambda_ = lambda_
        self.min_ratings = min_ratings
        self.B = None
        self.item_ids = None
        self.item_to_idx = {}

    def fit(self, X, item_ids):
        n_per_item = np.asarray((X > 0).sum(axis=0)).ravel()
        mask   = n_per_item >= self.min_ratings
        n_kept = int(mask.sum())
        if n_kept == 0:
            warnings.warn('EASE.fit: no items passed min_ratings filter.')
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


def ndcg_at_k(recommended, relevant_ratings, k=10):
    if not relevant_ratings:
        return 0.0
    dcg = sum(
        (relevant_ratings[item] / 5.0) / np.log2(rank + 2)
        for rank, item in enumerate(recommended[:k])
        if item in relevant_ratings
    )
    ideal = sorted(relevant_ratings.values(), reverse=True)[:k]
    idcg  = sum((r / 5.0) / np.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def bootstrap_ci(values, n_boot=N_BOOTSTRAP, ci=0.95):
    if not values:
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
    return X, np.array(all_item_ids, dtype=np.int64), user_to_row


if __name__ == '__main__':
    # load 2M sample (matches notebook 02_models.ipynb)
    print('Loading ratings...')
    ratings = pd.read_csv(DATA_DIR + 'ratings.csv')
    sample  = ratings.sample(n=N_SAMPLE, random_state=RANDOM_SEED)
    print(f'  Sample: {len(sample):,} ratings')

    # long-tail definition: bottom 80% of interaction volume
    print('Computing tail IDs...')
    film_pop = (
        ratings.groupby('movieId').size()
        .reset_index(name='cnt')
        .sort_values('cnt', ascending=False)
        .reset_index(drop=True)
    )
    film_pop['cumpct'] = film_pop['cnt'].cumsum() / film_pop['cnt'].sum() * 100
    cutoff   = (film_pop['cumpct'] >= 80).idxmax()
    tail_ids = set(film_pop.loc[cutoff + 1:, 'movieId'])
    all_movie_ids = set(ratings['movieId'].unique())
    print(f'  Total films: {len(all_movie_ids):,}  |  Tail films: {len(tail_ids):,}')
    del ratings; gc.collect()

    # train/test split
    train_df, test_df = train_test_split(sample, test_size=0.2, random_state=RANDOM_SEED)
    print(f'  Train: {len(train_df):,}  Test: {len(test_df):,}')

    # build sparse matrix and fit EASE
    print('Building sparse matrix...')
    X_train, item_ids_arr, _ = build_sparse_matrix(train_df)
    n_per_item = np.asarray((X_train > 0).sum(axis=0)).ravel()
    n_kept     = int((n_per_item >= MIN_RATINGS).sum())
    print(f'  Matrix: {X_train.shape}  |  Items passing filter: {n_kept:,}')
    print(f'  Gram matrix (float32): {n_kept**2*4/1e9:.2f} GB  peak: {n_kept**2*8/1e9:.2f} GB')

    print('Fitting EASE...')
    ease = EASE(lambda_=LAMBDA_EASE, min_ratings=MIN_RATINGS)
    ease.fit(X_train, item_ids_arr)
    del X_train; gc.collect()
    print(f'  B matrix: {len(ease.item_ids):,} items  ({ease.B.nbytes/1e6:.0f} MB)')

    # 500 eval users
    eval_users = list(set(train_df['userId'].unique()) & set(test_df['userId'].unique()))
    random.seed(RANDOM_SEED)
    eval_sample = random.sample(eval_users, min(N_EVAL_USERS, len(eval_users)))
    print(f'  Eval users: {len(eval_sample)}')

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
    train_seen_lookup = train_df.groupby('userId')['movieId'].apply(set).to_dict()

    # generate recommendations
    print('Generating recommendations...')
    all_recs = {}
    ndcg_vals = []
    for i, uid in enumerate(eval_sample):
        if i % 100 == 0:
            print(f'  ... {i}/{len(eval_sample)}', flush=True)
        relevant = test_lookup.get(uid, {})
        recs = ease.recommend(
            train_ratings_lookup.get(uid, {}),
            train_seen_lookup.get(uid, set()),
            n=TOP_N,
        )
        all_recs[uid] = recs
        if relevant:
            ndcg_vals.append(ndcg_at_k(recs, relevant))

    # NDCG@10
    mean_ndcg    = float(np.mean(ndcg_vals)) if ndcg_vals else np.nan
    ci_lo, ci_hi = bootstrap_ci(ndcg_vals)
    print(f'\n  NDCG@10 = {mean_ndcg:.4f}  CI=[{ci_lo:.4f}, {ci_hi:.4f}]')

    # coverage metrics
    rec_film_ids = {mid for recs in all_recs.values() for mid in recs}
    cat_cov  = len(rec_film_ids) / len(all_movie_ids) * 100
    lt_cov   = len(rec_film_ids & tail_ids) / len(tail_ids) * 100
    print(f'  Catalogue Coverage : {cat_cov:.4f}%  ({len(rec_film_ids):,} unique films)')
    print(f'  Long-tail Coverage : {lt_cov:.4f}%  ({len(rec_film_ids & tail_ids):,} tail films)')

    # update hybrid_summary.csv
    summary_path = OUTPUT_DIR + 'hybrid_summary.csv'
    summary = pd.read_csv(summary_path)

    if 'ease' in summary['model'].values:
        summary.loc[summary['model'] == 'ease', 'ndcg10'] = round(mean_ndcg, 4)
        summary.loc[summary['model'] == 'ease', 'ci_lo']  = round(ci_lo, 4)
        summary.loc[summary['model'] == 'ease', 'ci_hi']  = round(ci_hi, 4)
        summary.loc[summary['model'] == 'ease', 'cat_cov'] = round(cat_cov, 4)
        summary.loc[summary['model'] == 'ease', 'lt_cov']  = round(lt_cov, 4)
    else:
        new_row = pd.DataFrame([{
            'model':   'ease',
            'alpha':   None,
            'ndcg10':  round(mean_ndcg, 4),
            'ci_lo':   round(ci_lo, 4),
            'ci_hi':   round(ci_hi, 4),
            'cat_cov': round(cat_cov, 4),
            'lt_cov':  round(lt_cov, 4),
        }])
        summary = pd.concat([summary, new_row], ignore_index=True)

    summary.to_csv(summary_path, index=False)
    print(f'\nUpdated {summary_path}')
    print(summary.to_string(index=False))

    # regenerate HTML visualizations
    print('\nRegenerating HTML visualizations...')
    result = subprocess.run(
        ['python', 'scripts/update_viz_ease.py'],
        capture_output=True, text=True,
        cwd='/Users/kazuma/Desktop/film data analysis/popularity-bias-project',
    )
    print(result.stdout)
    if result.returncode != 0:
        print('STDERR:', result.stderr[:500])

    print('All done.')
