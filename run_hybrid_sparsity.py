"""
Experiment A with hybrid model (α=0.6) across sparsity levels.
Saves outputs/sparsity_expA_all_models.csv
"""

import numpy as np
import pandas as pd
import random, warnings, os, joblib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy import stats
from tqdm import tqdm

from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split as surprise_split

DATA_DIR   = '../ml-25m/'
OUTPUT_DIR = 'outputs/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_SEED    = 42
N_EVAL_USERS   = 500
N_BOOTSTRAP    = 1_000
TOP_N          = 10
LAMBDA_PENALTY = 0.3
ALPHA          = 0.6          # hybrid mixing weight (content-based share)

sparsity_levels = [1.0, 0.5, 0.1, 0.01, 0.001, 0.0001]
MODEL_NAMES = ['popularity_baseline', 'svd', 'penalized_svd',
               'content_based', 'hybrid']

# load data
print('Loading full ratings dataset (25M rows)...')
full_ratings = pd.read_csv(DATA_DIR + 'ratings.csv')
print(f'  Loaded {len(full_ratings):,} ratings')

print('Loading content-based model...')
cb_art        = joblib.load(OUTPUT_DIR + 'content_based_model.pkl')
tfidf_matrix  = cb_art['tfidf_matrix']   # (62423, 10000) sparse, L2-normalised
movie_ids_cb  = cb_art['movie_ids']       # numpy array of raw movieIds
mid_to_idx_cb = cb_art['movie_id_to_idx'] # {movieId: row_index}
print(f'  TF-IDF matrix: {tfidf_matrix.shape}')

# helper functions

def ndcg_at_k(recommended, relevant_ratings, k=10):
    if not relevant_ratings:
        return 0.0
    dcg = sum(
        (relevant_ratings[item] / 5.0) / np.log2(rank + 2)
        for rank, item in enumerate(recommended[:k])
        if item in relevant_ratings
    )
    ideal = sorted(relevant_ratings.values(), reverse=True)[:k]
    idcg  = sum((r / 5.0) / np.log2(rank + 2) for rank, r in enumerate(ideal))
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


def content_based_recs(user_id, train_df, tfidf_mat, mids, mid_to_idx, n=TOP_N):
    user_df  = train_df[train_df['userId'] == user_id]
    if user_df.empty:
        return []
    seen_ids = set(user_df['movieId'].astype(int))
    profile  = np.zeros(tfidf_mat.shape[1], dtype=np.float32)
    for _, row in user_df.iterrows():
        mid = int(row['movieId'])
        if mid in mid_to_idx:
            profile += (float(row['rating']) / 5.0) * \
                       tfidf_mat[mid_to_idx[mid]].toarray().ravel()
    norm = np.linalg.norm(profile)
    if norm == 0:
        return []
    profile /= norm
    sims   = tfidf_mat.dot(profile)
    ranked = sorted(
        [(int(mids[i]), float(sims[i]))
         for i in range(len(mids)) if int(mids[i]) not in seen_ids],
        key=lambda x: x[1], reverse=True
    )
    return [mid for mid, _ in ranked[:n]]


def build_svd_lookup(trainset):
    inner_to_raw = np.array([int(trainset.to_raw_iid(i))
                              for i in range(trainset.n_items)], dtype=np.int64)
    raw_uid_to_inner = {int(trainset.to_raw_uid(u)): u
                        for u in range(trainset.n_users)}
    return inner_to_raw, raw_uid_to_inner


def svd_scores_for_user(raw_uid, svd_model, inner_to_raw, raw_uid_to_inner):
    if raw_uid not in raw_uid_to_inner:
        return None, None
    inner_uid = raw_uid_to_inner[raw_uid]
    ts = svd_model.trainset
    scores = (ts.global_mean
              + svd_model.bu[inner_uid]
              + svd_model.bi
              + svd_model.qi.dot(svd_model.pu[inner_uid]))
    return inner_to_raw, scores   # both length n_items


# main loop
rows = []

for sparsity in tqdm(sparsity_levels, desc='Sparsity'):

    # 1. Subsample (Experiment A: uniform random → shrinks catalogue too)
    n_sample = max(int(len(full_ratings) * sparsity), 200)
    subset   = full_ratings.sample(n=n_sample, random_state=RANDOM_SEED)
    n_actual = len(subset)

    print(f'\n{"="*65}')
    print(f'SPARSITY = {sparsity}  ({n_actual:,} ratings, '
          f'{subset["movieId"].nunique():,} unique movies)')
    print(f'{"="*65}')

    # 2. Surprise split
    try:
        reader   = Reader(rating_scale=(0.5, 5.0))
        sur_data = Dataset.load_from_df(
            subset[['userId', 'movieId', 'rating']], reader
        )
        trainset, testset = surprise_split(
            sur_data, test_size=0.2, random_state=RANDOM_SEED
        )
    except Exception as exc:
        warnings.warn(f'  Split failed at sparsity {sparsity}: {exc}')
        continue

    # 3. Extract DataFrames
    trainset_df = pd.DataFrame(
        [(int(trainset.to_raw_uid(u)), int(trainset.to_raw_iid(i)), float(r))
         for u, i, r in trainset.all_ratings()],
        columns=['userId', 'movieId', 'rating']
    )
    testset_df = pd.DataFrame(
        [(int(uid), int(iid), float(r)) for uid, iid, r in testset],
        columns=['userId', 'movieId', 'rating']
    )

    # 4. Popularity stats
    film_pop = (
        trainset_df.groupby('movieId')
        .agg(rating_count=('rating', 'count'))
        .reset_index()
        .sort_values('rating_count', ascending=False)
        .reset_index(drop=True)
    )
    log_c    = np.log1p(film_pop.set_index('movieId')['rating_count'])
    pop_dict = (log_c / log_c.max()).to_dict() if len(log_c) > 0 else {}
    pop_top10 = film_pop.head(TOP_N)['movieId'].tolist()

    # 5. Train SVD
    svd_model = None
    rmse_val  = np.nan
    inner_to_raw = raw_uid_to_inner = None

    try:
        svd_model = SVD(n_factors=100, random_state=RANDOM_SEED)
        svd_model.fit(trainset)
        svd_preds = svd_model.test(testset)
        rmse_val  = accuracy.rmse(svd_preds, verbose=False)
        print(f'  SVD RMSE={rmse_val:.4f}  ({trainset.n_items:,} items)')
        inner_to_raw, raw_uid_to_inner = build_svd_lookup(trainset)
    except Exception as exc:
        warnings.warn(f'  SVD failed at sparsity {sparsity}: {exc}')

    # 6. Evaluation users
    train_users = set(trainset_df['userId'].unique())
    test_users  = set(testset_df['userId'].unique())
    eval_users  = list(train_users & test_users)
    random.seed(RANDOM_SEED)
    eval_sample = random.sample(eval_users, min(N_EVAL_USERS, len(eval_users)))
    print(f'  Eval users: {len(eval_sample)}')

    test_lookup  = (testset_df.groupby('userId')
                    .apply(lambda x: dict(zip(x['movieId'], x['rating'])),
                           include_groups=False)
                    .to_dict())
    train_lookup = trainset_df.groupby('userId')['movieId'].apply(set).to_dict()

    # 7. Recommend and score
    user_ndcg = {m: [] for m in MODEL_NAMES}

    for uid in eval_sample:
        relevant = test_lookup.get(uid, {})
        if not relevant:
            continue
        seen = train_lookup.get(uid, set())

        # popularity baseline
        pb = [m for m in pop_top10 if m not in seen]
        user_ndcg['popularity_baseline'].append(ndcg_at_k(pb, relevant))

        # SVD + penalized (batch)
        svd_raw_iids = svd_scores = None
        if svd_model is not None:
            svd_raw_iids, svd_scores = svd_scores_for_user(
                uid, svd_model, inner_to_raw, raw_uid_to_inner
            )

        if svd_scores is not None:
            # standard SVD
            ranked_svd = sorted(
                [(int(svd_raw_iids[i]), float(svd_scores[i]))
                 for i in range(len(svd_raw_iids))
                 if int(svd_raw_iids[i]) not in seen],
                key=lambda x: x[1], reverse=True
            )
            recs_svd = [m for m, _ in ranked_svd[:TOP_N]]
            user_ndcg['svd'].append(ndcg_at_k(recs_svd, relevant))

            # penalized SVD
            pop_arr  = np.array([pop_dict.get(int(svd_raw_iids[i]), 0.0)
                                  for i in range(len(svd_raw_iids))],
                                 dtype=np.float32)
            adj      = svd_scores * (1.0 - LAMBDA_PENALTY * pop_arr)
            ranked_pen = sorted(
                [(int(svd_raw_iids[i]), float(adj[i]))
                 for i in range(len(svd_raw_iids))
                 if int(svd_raw_iids[i]) not in seen],
                key=lambda x: x[1], reverse=True
            )
            recs_pen = [m for m, _ in ranked_pen[:TOP_N]]
            user_ndcg['penalized_svd'].append(ndcg_at_k(recs_pen, relevant))

            # normalised SVD scores dict {movieId: svd_norm} for hybrid
            svd_norm_dict = {
                int(svd_raw_iids[i]): (float(svd_scores[i]) - 0.5) / 4.5
                for i in range(len(svd_raw_iids))
            }
        else:
            user_ndcg['svd'].append(0.0)
            user_ndcg['penalized_svd'].append(0.0)
            svd_norm_dict = {}

        # content-based
        user_train_df = trainset_df[trainset_df['userId'] == uid]
        cb_recs = content_based_recs(
            uid, user_train_df, tfidf_matrix, movie_ids_cb, mid_to_idx_cb
        )
        user_ndcg['content_based'].append(ndcg_at_k(cb_recs, relevant))

        # content-based scores dict {movieId: cosine_sim} for hybrid
        if not user_train_df.empty:
            profile = np.zeros(tfidf_matrix.shape[1], dtype=np.float32)
            for _, row in user_train_df.iterrows():
                mid = int(row['movieId'])
                if mid in mid_to_idx_cb:
                    profile += (float(row['rating']) / 5.0) * \
                               tfidf_matrix[mid_to_idx_cb[mid]].toarray().ravel()
            norm = np.linalg.norm(profile)
            if norm > 0:
                profile /= norm
                sims = tfidf_matrix.dot(profile)   # shape (62423,)
                cb_score_dict = {
                    int(movie_ids_cb[i]): float(sims[i])
                    for i in range(len(movie_ids_cb))
                }
            else:
                cb_score_dict = {}
        else:
            cb_score_dict = {}

        # hybrid (α=0.6): score over union of SVD + CB candidate pools
        if svd_norm_dict or cb_score_dict:
            all_cands = (set(svd_norm_dict.keys()) | set(cb_score_dict.keys())) - seen
            hybrid_scores = {
                mid: (1 - ALPHA) * svd_norm_dict.get(mid, 0.0)
                     + ALPHA     * cb_score_dict.get(mid, 0.0)
                for mid in all_cands
            }
            ranked_hyb = sorted(hybrid_scores.items(), key=lambda x: x[1],
                                reverse=True)
            recs_hyb = [m for m, _ in ranked_hyb[:TOP_N]]
        else:
            recs_hyb = []
        user_ndcg['hybrid'].append(ndcg_at_k(recs_hyb, relevant))

    # 8. Aggregate
    for model in MODEL_NAMES:
        vals = np.array(user_ndcg[model])
        mean_ndcg    = float(np.mean(vals)) if len(vals) > 0 else np.nan
        ci_lo, ci_hi = bootstrap_ci(vals)
        rows.append({
            'sparsity' : sparsity,
            'n_ratings': n_actual,
            'model'    : model,
            'ndcg'     : round(mean_ndcg, 4),
            'ci_lower' : round(ci_lo,     4),
            'ci_upper' : round(ci_hi,     4),
        })
        print(f'  [{model:<22}]  ndcg={mean_ndcg:.4f}  '
              f'[{ci_lo:.4f}, {ci_hi:.4f}]')

print('\n✓ All sparsity levels complete.')

# save and print
results_df = pd.DataFrame(rows)
out_csv    = OUTPUT_DIR + 'sparsity_expA_all_models.csv'
results_df.to_csv(out_csv, index=False)
print(f'\nSaved  →  {out_csv}')
print(f'Shape  :  {results_df.shape}')
print()
print(results_df.to_string(index=False))

# figure: NDCG@10 curves including hybrid
COLOURS = {
    'popularity_baseline': '#999999',
    'svd'               : '#1f77b4',
    'penalized_svd'     : '#d62728',
    'content_based'     : '#2ca02c',
    'hybrid'            : '#ff7f0e',
}
MARKERS = {
    'popularity_baseline': 's',
    'svd'               : 'o',
    'penalized_svd'     : '^',
    'content_based'     : 'D',
    'hybrid'            : 'P',
}
LABELS = {
    'popularity_baseline': 'Popularity Baseline',
    'svd'               : 'Standard SVD',
    'penalized_svd'     : 'Penalised SVD (λ=0.3)',
    'content_based'     : 'Content-Based',
    'hybrid'            : f'Hybrid (α={ALPHA})',
}

fig, ax = plt.subplots(figsize=(10, 5))
for model in MODEL_NAMES:
    sub = results_df[results_df['model'] == model].sort_values('sparsity')
    if sub.empty:
        continue
    x, y     = sub['sparsity'].values, sub['ndcg'].values
    y_lo, y_hi = sub['ci_lower'].values, sub['ci_upper'].values
    ax.plot(x, y, color=COLOURS[model], marker=MARKERS[model],
            label=LABELS[model], linewidth=2, markersize=7)
    ax.fill_between(x, y_lo, y_hi, color=COLOURS[model], alpha=0.15)

ax.set_xscale('log')
ax.set_xlabel('Sparsity Level (fraction of full dataset)', fontsize=12)
ax.set_ylabel('NDCG@10', fontsize=12)
ax.set_title('Experiment A + Hybrid: NDCG@10 vs. Dataset Sparsity\n'
             '(shaded = 95% bootstrap CI, n=500 users)', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()

fig_path = OUTPUT_DIR + 'sparsity_ndcg_hybrid.png'
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved  →  {fig_path}')

print('\n✓ run_hybrid_sparsity.py complete.')
