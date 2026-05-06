"""
Experiment B with hybrid model (α=0.6) — fixed 62,423-film catalogue.
Saves outputs/sparsity_expB_all_models.csv and sparsity_expAB_combined.csv
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
ALPHA          = 0.6

sparsity_levels = [1.0, 0.5, 0.1, 0.01, 0.001, 0.0001]
MODEL_NAMES = ['popularity_baseline', 'svd', 'penalized_svd',
               'content_based', 'hybrid']

# load data
print('Loading full ratings dataset (25M rows)...')
full_ratings = pd.read_csv(DATA_DIR + 'ratings.csv')
print(f'  Loaded {len(full_ratings):,} ratings')

print('Loading content-based model...')
cb_art        = joblib.load(OUTPUT_DIR + 'content_based_model.pkl')
tfidf_matrix  = cb_art['tfidf_matrix']
movie_ids_cb  = cb_art['movie_ids']
mid_to_idx_cb = cb_art['movie_id_to_idx']
print(f'  TF-IDF matrix: {tfidf_matrix.shape}')

FULL_CATALOGUE_SIZE = tfidf_matrix.shape[0]   # 62,423

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
    a = (1 - ci) / 2
    return (float(np.percentile(boot, a * 100)),
            float(np.percentile(boot, (1 - a) * 100)))


def fixed_catalogue_sample(df, sparsity, random_seed=RANDOM_SEED):
    """Keep max(1, floor(n_i * sparsity)) ratings per movie."""
    return (
        df.groupby('movieId', group_keys=False)
        .apply(lambda x: x.sample(
            n=max(1, int(len(x) * sparsity)),
            random_state=random_seed
        ))
        .reset_index(drop=True)
    )


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
    return inner_to_raw, scores


def content_based_recs(user_id, train_df, tfidf_mat, mids, mid_to_idx, n=TOP_N):
    user_df = train_df[train_df['userId'] == user_id]
    if user_df.empty:
        return [], {}
    seen_ids = set(user_df['movieId'].astype(int))
    profile  = np.zeros(tfidf_mat.shape[1], dtype=np.float32)
    for _, row in user_df.iterrows():
        mid = int(row['movieId'])
        if mid in mid_to_idx:
            profile += (float(row['rating']) / 5.0) * \
                       tfidf_mat[mid_to_idx[mid]].toarray().ravel()
    norm = np.linalg.norm(profile)
    if norm == 0:
        return [], {}
    profile /= norm
    sims = tfidf_mat.dot(profile)
    score_dict = {int(mids[i]): float(sims[i]) for i in range(len(mids))
                  if int(mids[i]) not in seen_ids}
    ranked = sorted(score_dict.items(), key=lambda x: x[1], reverse=True)
    return [mid for mid, _ in ranked[:n]], score_dict


# fixed tail IDs from full 25M dataset
print('\nComputing fixed tail IDs from full dataset...')
_fp_full = (
    full_ratings.groupby('movieId')
    .agg(rating_count=('rating', 'count'))
    .reset_index()
    .sort_values('rating_count', ascending=False)
    .reset_index(drop=True)
)
_fp_full['cumpct'] = (_fp_full['rating_count'].cumsum() /
                      _fp_full['rating_count'].sum() * 100)
_cutoff      = (_fp_full['cumpct'] >= 80).idxmax()
tail_ids_B   = set(_fp_full.loc[_cutoff + 1:, 'movieId'])
print(f'  Full catalogue : {FULL_CATALOGUE_SIZE:,} films')
print(f'  Tail (fixed)   : {len(tail_ids_B):,} films')

# main loop
rows_B = []

for sparsity in tqdm(sparsity_levels, desc='Exp B sparsity'):

    subset_B = fixed_catalogue_sample(full_ratings, sparsity)
    n_actual = len(subset_B)
    avg_rpm  = n_actual / subset_B['movieId'].nunique()

    print(f'\n{"="*65}')
    print(f'EXPERIMENT B  |  SPARSITY = {sparsity}')
    print(f'{"="*65}')
    print(f'  Total ratings : {n_actual:,}   '
          f'Avg ratings/movie : {avg_rpm:.2f}   '
          f'Unique movies : {subset_B["movieId"].nunique():,}')

    # surprise split
    try:
        reader   = Reader(rating_scale=(0.5, 5.0))
        sur_data = Dataset.load_from_df(
            subset_B[['userId', 'movieId', 'rating']], reader
        )
        trainset_B, testset_B = surprise_split(
            sur_data, test_size=0.2, random_state=RANDOM_SEED
        )
    except Exception as exc:
        warnings.warn(f'  Split failed: {exc}')
        continue

    # extract DataFrames
    trainset_df_B = pd.DataFrame(
        [(int(trainset_B.to_raw_uid(u)), int(trainset_B.to_raw_iid(i)), float(r))
         for u, i, r in trainset_B.all_ratings()],
        columns=['userId', 'movieId', 'rating']
    )
    testset_df_B = pd.DataFrame(
        [(int(uid), int(iid), float(r)) for uid, iid, r in testset_B],
        columns=['userId', 'movieId', 'rating']
    )

    # popularity stats
    film_pop_B = (
        trainset_df_B.groupby('movieId')
        .agg(rating_count=('rating', 'count'))
        .reset_index()
        .sort_values('rating_count', ascending=False)
        .reset_index(drop=True)
    )
    log_c_B    = np.log1p(film_pop_B.set_index('movieId')['rating_count'])
    pop_dict_B = (log_c_B / log_c_B.max()).to_dict() if len(log_c_B) > 0 else {}
    pop_top10_B = film_pop_B.head(TOP_N)['movieId'].tolist()

    # train SVD
    svd_model_B = None
    inner_to_raw_B = raw_uid_to_inner_B = None

    try:
        svd_model_B = SVD(n_factors=100, random_state=RANDOM_SEED)
        svd_model_B.fit(trainset_B)
        svd_preds_B = svd_model_B.test(testset_B)
        rmse_B      = accuracy.rmse(svd_preds_B, verbose=False)
        print(f'  SVD RMSE={rmse_B:.4f}  ({trainset_B.n_items:,} items)')
        inner_to_raw_B, raw_uid_to_inner_B = build_svd_lookup(trainset_B)
    except Exception as exc:
        warnings.warn(f'  SVD failed: {exc}')

    # evaluation users
    train_users_B = set(trainset_df_B['userId'].unique())
    test_users_B  = set(testset_df_B['userId'].unique())
    eval_users_B  = list(train_users_B & test_users_B)
    random.seed(RANDOM_SEED)
    eval_sample_B = random.sample(eval_users_B,
                                   min(N_EVAL_USERS, len(eval_users_B)))
    print(f'  Eval users: {len(eval_sample_B)}')

    test_lookup_B  = (testset_df_B.groupby('userId')
                      .apply(lambda x: dict(zip(x['movieId'], x['rating'])),
                             include_groups=False)
                      .to_dict())
    train_lookup_B = trainset_df_B.groupby('userId')['movieId'].apply(set).to_dict()

    # recommend and score
    user_ndcg_B = {m: [] for m in MODEL_NAMES}

    for uid in eval_sample_B:
        relevant = test_lookup_B.get(uid, {})
        if not relevant:
            continue
        seen = train_lookup_B.get(uid, set())

        # popularity baseline
        pb = [m for m in pop_top10_B if m not in seen]
        user_ndcg_B['popularity_baseline'].append(ndcg_at_k(pb, relevant))

        # SVD batch
        svd_raw_iids = svd_sc = None
        if svd_model_B is not None:
            svd_raw_iids, svd_sc = svd_scores_for_user(
                uid, svd_model_B, inner_to_raw_B, raw_uid_to_inner_B
            )

        if svd_sc is not None:
            ranked_svd = sorted(
                [(int(svd_raw_iids[i]), float(svd_sc[i]))
                 for i in range(len(svd_raw_iids))
                 if int(svd_raw_iids[i]) not in seen],
                key=lambda x: x[1], reverse=True
            )
            recs_svd = [m for m, _ in ranked_svd[:TOP_N]]
            user_ndcg_B['svd'].append(ndcg_at_k(recs_svd, relevant))

            pop_arr_B = np.array([pop_dict_B.get(int(svd_raw_iids[i]), 0.0)
                                   for i in range(len(svd_raw_iids))],
                                  dtype=np.float32)
            adj = svd_sc * (1.0 - LAMBDA_PENALTY * pop_arr_B)
            ranked_pen = sorted(
                [(int(svd_raw_iids[i]), float(adj[i]))
                 for i in range(len(svd_raw_iids))
                 if int(svd_raw_iids[i]) not in seen],
                key=lambda x: x[1], reverse=True
            )
            recs_pen = [m for m, _ in ranked_pen[:TOP_N]]
            user_ndcg_B['penalized_svd'].append(ndcg_at_k(recs_pen, relevant))

            svd_norm_dict = {
                int(svd_raw_iids[i]): (float(svd_sc[i]) - 0.5) / 4.5
                for i in range(len(svd_raw_iids))
            }
        else:
            user_ndcg_B['svd'].append(0.0)
            user_ndcg_B['penalized_svd'].append(0.0)
            svd_norm_dict = {}

        # content-based (returns recs list + full score dict for hybrid)
        user_train_df_B = trainset_df_B[trainset_df_B['userId'] == uid]
        cb_recs, cb_score_dict = content_based_recs(
            uid, user_train_df_B, tfidf_matrix, movie_ids_cb, mid_to_idx_cb
        )
        user_ndcg_B['content_based'].append(ndcg_at_k(cb_recs, relevant))

        # hybrid
        if svd_norm_dict or cb_score_dict:
            all_cands = (set(svd_norm_dict.keys()) | set(cb_score_dict.keys())) - seen
            hybrid_sc = {
                mid: (1 - ALPHA) * svd_norm_dict.get(mid, 0.0)
                     + ALPHA     * cb_score_dict.get(mid, 0.0)
                for mid in all_cands
            }
            ranked_hyb = sorted(hybrid_sc.items(), key=lambda x: x[1], reverse=True)
            recs_hyb   = [m for m, _ in ranked_hyb[:TOP_N]]
        else:
            recs_hyb = []
        user_ndcg_B['hybrid'].append(ndcg_at_k(recs_hyb, relevant))

    # aggregate
    for model in MODEL_NAMES:
        vals = np.array(user_ndcg_B[model])
        mean_ndcg    = float(np.mean(vals)) if len(vals) > 0 else np.nan
        ci_lo, ci_hi = bootstrap_ci(vals)
        rows_B.append({
            'experiment': 'B',
            'sparsity'  : sparsity,
            'n_ratings' : n_actual,
            'model'     : model,
            'ndcg'      : round(mean_ndcg, 4),
            'ci_lower'  : round(ci_lo,     4),
            'ci_upper'  : round(ci_hi,     4),
        })
        print(f'  [{model:<22}]  ndcg={mean_ndcg:.4f}  '
              f'[{ci_lo:.4f}, {ci_hi:.4f}]')

print('\n✓ Experiment B main loop complete.')

# save Experiment B results
df_B = pd.DataFrame(rows_B)
out_B = OUTPUT_DIR + 'sparsity_expB_all_models.csv'
df_B.to_csv(out_B, index=False)
print(f'Saved  →  {out_B}')

# combine A + B and save to outputs/sparsity_expAB_combined.csv
out_A = OUTPUT_DIR + 'sparsity_expA_all_models.csv'

if os.path.exists(out_A):
    df_A = pd.read_csv(out_A)
    # add experiment column if not already present
    if 'experiment' not in df_A.columns:
        df_A.insert(0, 'experiment', 'A')
    combined = pd.concat([df_A, df_B], ignore_index=True)
    print('\nCombined A + B:')
else:
    # A not yet done — save B only with a note
    combined = df_B.copy()
    print('\nExperiment A results not yet available — saving B only.')
    print('Re-run this script after run_hybrid_sparsity.py completes to get A+B.')

combined_path = OUTPUT_DIR + 'sparsity_expAB_combined.csv'
combined.to_csv(combined_path, index=False)
print(f'Saved  →  {combined_path}')
print(f'Shape  :  {combined.shape}')
print()
print(combined.to_string(index=False))

print('\n✓ run_hybrid_sparsity_B.py complete.')
