"""
Experiment B: fixed catalogue, varying rating density.
"""

# imports
import numpy as np
import pandas as pd
import random
import warnings
import os
import joblib

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

# load data
print('Loading full ratings dataset (25M rows)...')
full_ratings = pd.read_csv(DATA_DIR + 'ratings.csv')
print(f'  Loaded {len(full_ratings):,} ratings')

print('Loading content-based model...')
cb_artefact   = joblib.load(OUTPUT_DIR + 'content_based_model.pkl')
tfidf_matrix  = cb_artefact['tfidf_matrix']
movie_ids_cb  = cb_artefact['movie_ids']
mid_to_idx_cb = cb_artefact['movie_id_to_idx']
print(f'  TF-IDF matrix: {tfidf_matrix.shape}')

FULL_CATALOGUE_SIZE = tfidf_matrix.shape[0]   # 62,423
MODEL_NAMES = ['popularity_baseline', 'svd', 'penalized_svd', 'content_based']

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


# experiment B helpers

def fixed_catalogue_sample(df, sparsity, random_seed=RANDOM_SEED):
    """Sample a fraction of each movie's ratings (min 1 per movie)."""
    return (
        df.groupby('movieId', group_keys=False)
        .apply(lambda x: x.sample(
            n=max(1, int(len(x) * sparsity)),
            random_state=random_seed
        ))
        .reset_index(drop=True)
    )


def build_inner_to_raw(trainset):
    inner_to_raw = np.array([int(trainset.to_raw_iid(i))
                              for i in range(trainset.n_items)], dtype=np.int64)
    raw_to_inner = {int(trainset.to_raw_iid(i)): i
                    for i in range(trainset.n_items)}
    raw_uid_to_inner = {int(trainset.to_raw_uid(u)): u
                        for u in range(trainset.n_users)}
    return inner_to_raw, raw_to_inner, raw_uid_to_inner


def svd_recs_batch(user_id, svd_model, inner_to_raw_arr, raw_uid_to_inner,
                   seen_ids, n=TOP_N):
    raw_uid = int(user_id)
    if raw_uid not in raw_uid_to_inner:
        return []
    inner_uid = raw_uid_to_inner[raw_uid]
    ts = svd_model.trainset
    scores = (ts.global_mean
              + svd_model.bu[inner_uid]
              + svd_model.bi
              + svd_model.qi.dot(svd_model.pu[inner_uid]))
    ranked  = sorted(
        [(int(inner_to_raw_arr[i]), float(scores[i]))
         for i in range(ts.n_items) if int(inner_to_raw_arr[i]) not in seen_ids],
        key=lambda x: x[1], reverse=True
    )
    return [mid for mid, _ in ranked[:n]]


def penalized_svd_recs_batch(user_id, svd_model, inner_to_raw_arr,
                              raw_uid_to_inner, seen_ids, pop_arr,
                              lam=LAMBDA_PENALTY, n=TOP_N):
    raw_uid = int(user_id)
    if raw_uid not in raw_uid_to_inner:
        return []
    inner_uid = raw_uid_to_inner[raw_uid]
    ts = svd_model.trainset
    scores = (ts.global_mean
              + svd_model.bu[inner_uid]
              + svd_model.bi
              + svd_model.qi.dot(svd_model.pu[inner_uid]))
    adj_scores = scores * (1.0 - lam * pop_arr)
    ranked  = sorted(
        [(int(inner_to_raw_arr[i]), float(adj_scores[i]))
         for i in range(ts.n_items) if int(inner_to_raw_arr[i]) not in seen_ids],
        key=lambda x: x[1], reverse=True
    )
    return [mid for mid, _ in ranked[:n]]


def catalog_coverage_B(recs_dict, full_size=FULL_CATALOGUE_SIZE):
    recommended = {mid for lst in recs_dict.values() for mid in lst}
    return len(recommended) / full_size * 100


def longtail_coverage_B(recs_dict, tail_ids):
    if not tail_ids:
        return 0.0
    recommended = {mid for lst in recs_dict.values() for mid in lst}
    return len(recommended & tail_ids) / len(tail_ids) * 100


# fixed tail IDs (derived once from full 25M dataset)
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
_cutoff_full   = (_fp_full['cumpct'] >= 80).idxmax()
tail_ids_fixed = set(_fp_full.loc[_cutoff_full + 1:, 'movieId'])
head_ids_fixed = set(_fp_full.loc[:_cutoff_full, 'movieId'])
print(f'  Full catalogue : {FULL_CATALOGUE_SIZE:,} films')
print(f'  Head (top 80%) : {len(head_ids_fixed):,} films')
print(f'  Tail           : {len(tail_ids_fixed):,} films')

# experiment B main loop
sparsity_levels = [1.0, 0.5, 0.1, 0.01, 0.001, 0.0001]
results_rows_B  = []
per_user_ndcg_B = {}

for sparsity in tqdm(sparsity_levels, desc='Exp B sparsity'):

    # 1. Fixed-catalogue subsample
    subset_B = fixed_catalogue_sample(full_ratings, sparsity)

    n_unique_movies_B = subset_B['movieId'].nunique()
    n_unique_users_B  = subset_B['userId'].nunique()
    avg_ratings_movie = len(subset_B) / n_unique_movies_B

    print(f'\n{"="*65}')
    print(f'EXPERIMENT B  |  SPARSITY = {sparsity}')
    print(f'{"="*65}')
    print(f'  Total ratings in subsample : {len(subset_B):>10,}')
    print(f'  Unique users               : {n_unique_users_B:>10,}')
    print(f'  Unique movies in subsample : {n_unique_movies_B:>10,}  '
          f'(target: all {FULL_CATALOGUE_SIZE:,})')
    print(f'  Avg ratings per movie      : {avg_ratings_movie:>10.2f}')
    print(f'  CB model scope             : {FULL_CATALOGUE_SIZE:>10,}  (fixed)')

    # 2. Surprise split
    try:
        reader_B   = Reader(rating_scale=(0.5, 5.0))
        sur_data_B = Dataset.load_from_df(
            subset_B[['userId', 'movieId', 'rating']], reader_B
        )
        trainset_B, testset_B = surprise_split(
            sur_data_B, test_size=0.2, random_state=RANDOM_SEED
        )
    except Exception as exc:
        warnings.warn(f'  [Exp B] Data split failed at sparsity {sparsity}: {exc}')
        continue

    # 3. Extract DataFrames
    trainset_df_B = pd.DataFrame(
        [(int(trainset_B.to_raw_uid(u)),
          int(trainset_B.to_raw_iid(i)),
          float(r))
         for u, i, r in trainset_B.all_ratings()],
        columns=['userId', 'movieId', 'rating'],
    )
    testset_df_B = pd.DataFrame(
        [(int(uid), int(iid), float(r)) for uid, iid, r in testset_B],
        columns=['userId', 'movieId', 'rating'],
    )

    n_svd_train_movies = trainset_df_B['movieId'].nunique()
    print(f'  SVD training movies        : {n_svd_train_movies:>10,}  '
          f'({n_svd_train_movies / FULL_CATALOGUE_SIZE * 100:.1f}% of full)')

    # 4. Popularity stats from training subset
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

    # 5. Train SVD
    svd_model_B       = None
    rmse_svd_B        = np.nan
    inner_to_raw_B    = None
    raw_uid_to_inner_B = None
    pop_arr_B         = None

    try:
        svd_model_B = SVD(n_factors=100, random_state=RANDOM_SEED)
        svd_model_B.fit(trainset_B)
        svd_preds_B = svd_model_B.test(testset_B)
        rmse_svd_B  = accuracy.rmse(svd_preds_B, verbose=False)
        print(f'  SVD RMSE={rmse_svd_B:.4f}  ({trainset_B.n_items:,} items)')

        inner_to_raw_B, _, raw_uid_to_inner_B = build_inner_to_raw(trainset_B)
        pop_arr_B = np.array(
            [pop_dict_B.get(int(inner_to_raw_B[i]), 0.0)
             for i in range(trainset_B.n_items)],
            dtype=np.float32
        )
    except Exception as exc:
        warnings.warn(f'  [Exp B] SVD failed at sparsity {sparsity}: {exc}')

    # 6. Evaluation users
    train_users_B = set(trainset_df_B['userId'].unique())
    test_users_B  = set(testset_df_B['userId'].unique())
    eval_users_B  = list(train_users_B & test_users_B)
    random.seed(RANDOM_SEED)
    eval_sample_B = random.sample(eval_users_B,
                                  min(N_EVAL_USERS, len(eval_users_B)))
    print(f'  Evaluation users: {len(eval_sample_B)}')

    test_lookup_B  = (
        testset_df_B.groupby('userId')
        .apply(lambda x: dict(zip(x['movieId'], x['rating'])), include_groups=False)
        .to_dict()
    )
    train_lookup_B = (
        trainset_df_B.groupby('userId')['movieId'].apply(set).to_dict()
    )

    # 7. Recommendations
    user_ndcg_B     = {m: [] for m in MODEL_NAMES}
    user_recs_all_B = {m: {} for m in MODEL_NAMES}

    for uid in eval_sample_B:
        relevant = test_lookup_B.get(uid, {})
        if not relevant:
            continue
        seen = train_lookup_B.get(uid, set())

        # popularity baseline
        pb = [m for m in pop_top10_B if m not in seen]
        user_ndcg_B['popularity_baseline'].append(ndcg_at_k(pb, relevant))
        user_recs_all_B['popularity_baseline'][uid] = pb

        # SVD (batch)
        if svd_model_B is not None and inner_to_raw_B is not None:
            try:
                sr = svd_recs_batch(uid, svd_model_B, inner_to_raw_B,
                                    raw_uid_to_inner_B, seen)
                user_ndcg_B['svd'].append(ndcg_at_k(sr, relevant))
                user_recs_all_B['svd'][uid] = sr

                pr = penalized_svd_recs_batch(uid, svd_model_B, inner_to_raw_B,
                                              raw_uid_to_inner_B, seen, pop_arr_B)
                user_ndcg_B['penalized_svd'].append(ndcg_at_k(pr, relevant))
                user_recs_all_B['penalized_svd'][uid] = pr
            except Exception as exc:
                warnings.warn(f'  [Exp B] SVD rec failed uid={uid}: {exc}')

        # content-based
        user_train_df_B = trainset_df_B[trainset_df_B['userId'] == uid]
        cb = content_based_recs(
            uid, user_train_df_B, tfidf_matrix, movie_ids_cb, mid_to_idx_cb
        )
        user_ndcg_B['content_based'].append(ndcg_at_k(cb, relevant))
        user_recs_all_B['content_based'][uid] = cb

    # 8 & 9. Aggregate + bootstrap CIs
    for model in MODEL_NAMES:
        ndcg_vals = np.array(user_ndcg_B[model])
        recs_dict = user_recs_all_B[model]

        mean_ndcg    = float(np.mean(ndcg_vals)) if len(ndcg_vals) > 0 else np.nan
        ci_lo, ci_hi = bootstrap_ci(ndcg_vals)
        cat_cov      = catalog_coverage_B(recs_dict)
        lt_cov       = longtail_coverage_B(recs_dict, tail_ids_fixed)
        rmse_val     = rmse_svd_B if model in ('svd', 'penalized_svd') else np.nan

        results_rows_B.append({
            'sparsity_level'       : sparsity,
            'model'                : model,
            'avg_ratings_per_movie': round(avg_ratings_movie, 2),
            'rmse'                 : round(rmse_val, 4) if not np.isnan(rmse_val) else np.nan,
            'ndcg10'               : round(mean_ndcg, 4),
            'ndcg10_ci_lower'      : round(ci_lo, 4),
            'ndcg10_ci_upper'      : round(ci_hi, 4),
            'catalogue_coverage'   : round(cat_cov, 4),
            'long_tail_coverage'   : round(lt_cov, 4),
        })
        per_user_ndcg_B[(sparsity, model)] = ndcg_vals.tolist()

        print(f'  [{model:<22}]  ndcg10={mean_ndcg:.4f}  '
              f'cat_cov={cat_cov:.2f}%  lt_cov={lt_cov:.2f}%')

print('\n✓ Experiment B main loop complete.')

# save results
results_df_B = pd.DataFrame(results_rows_B)
out_csv_B    = OUTPUT_DIR + 'sparsity_expB_no_hybrid.csv'
results_df_B.to_csv(out_csv_B, index=False)
print(f'\nSaved  →  {out_csv_B}')
print(f'Shape  :  {results_df_B.shape}')
print()
print(results_df_B.to_string(index=False))

# plotting constants
COLOURS = {
    'popularity_baseline': '#999999',
    'svd'               : '#1f77b4',
    'penalized_svd'     : '#d62728',
    'content_based'     : '#2ca02c',
}
MARKERS = {
    'popularity_baseline': 's',
    'svd'               : 'o',
    'penalized_svd'     : '^',
    'content_based'     : 'D',
}
LABELS = {
    'popularity_baseline': 'Popularity Baseline',
    'svd'               : 'Standard SVD',
    'penalized_svd'     : 'Penalised SVD (λ=0.3)',
    'content_based'     : 'Content-Based',
}

# crossover analysis (Experiment B)
pivot_B = (results_df_B
           .pivot(index='sparsity_level', columns='model', values='ndcg10')
           .reset_index()
           .sort_values('sparsity_level'))

crossover_sparsity_B = None
crossover_row_B      = None

for _, row in pivot_B.iterrows():
    cb_val  = row.get('content_based', np.nan)
    pen_val = row.get('penalized_svd', np.nan)
    if pd.notna(cb_val) and pd.notna(pen_val) and cb_val > pen_val:
        crossover_sparsity_B = row['sparsity_level']
        crossover_row_B      = row
    else:
        if crossover_sparsity_B is not None:
            break

print('\n' + '=' * 70)
if crossover_sparsity_B is not None:
    n_crossover_B   = int(len(full_ratings) * crossover_sparsity_B)
    avg_at_crossover = results_df_B[
        results_df_B['sparsity_level'] == crossover_sparsity_B
    ]['avg_ratings_per_movie'].iloc[0]
    print('EXPERIMENT B — CROSSOVER THRESHOLD')
    print(f'  Sparsity level        : {crossover_sparsity_B}')
    print(f'  Approx. total ratings : {n_crossover_B:,}')
    print(f'  Avg ratings/movie     : {avg_at_crossover:.2f}')
    print(f'  Content-based NDCG@10 : {crossover_row_B["content_based"]:.4f}')
    print(f'  Penalised SVD  NDCG@10: {crossover_row_B["penalized_svd"]:.4f}')
else:
    print('EXPERIMENT B — No crossover found within evaluated sparsity range.')
print('=' * 70)

# figure 1: NDCG@10 curves (Experiment B)
fig, ax = plt.subplots(figsize=(10, 5))

for model in MODEL_NAMES:
    sub = results_df_B[results_df_B['model'] == model].sort_values('sparsity_level')
    if sub.empty:
        continue
    x, y = sub['sparsity_level'].values, sub['ndcg10'].values
    y_lo, y_hi = sub['ndcg10_ci_lower'].values, sub['ndcg10_ci_upper'].values
    ax.plot(x, y, color=COLOURS[model], marker=MARKERS[model],
            label=LABELS[model], linewidth=2, markersize=7)
    ax.fill_between(x, y_lo, y_hi, color=COLOURS[model], alpha=0.15)

if crossover_sparsity_B is not None:
    ax.axvline(crossover_sparsity_B, color='black', linestyle='--',
               linewidth=1.5,
               label=f'Crossover @ sparsity={crossover_sparsity_B}')

ax.set_xscale('log')
ax.set_xlabel('Sparsity Level (fraction of ratings per movie kept)', fontsize=12)
ax.set_ylabel('NDCG@10', fontsize=12)
ax.set_title('Experiment B: NDCG@10 vs. Rating Density\n'
             '(Fixed 62,423-film catalogue; shaded = 95% bootstrap CI, n=500 users)',
             fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()

ndcg_fig_B = OUTPUT_DIR + 'sparsity_expB_ndcg_curves.png'
plt.savefig(ndcg_fig_B, dpi=150, bbox_inches='tight')
plt.close()
print(f'\nSaved  →  {ndcg_fig_B}')

# figure 2: Coverage curves (Experiment B)
fig, ax = plt.subplots(figsize=(10, 5))

for model in MODEL_NAMES:
    sub = results_df_B[results_df_B['model'] == model].sort_values('sparsity_level')
    if sub.empty:
        continue
    ax.plot(sub['sparsity_level'], sub['catalogue_coverage'],
            color=COLOURS[model], marker=MARKERS[model],
            label=LABELS[model], linewidth=2, markersize=7)

if crossover_sparsity_B is not None:
    ax.axvline(crossover_sparsity_B, color='black', linestyle='--',
               linewidth=1.5, label=f'Crossover @ {crossover_sparsity_B}')

ax.set_xscale('log')
ax.set_xlabel('Sparsity Level (fraction of ratings per movie kept)', fontsize=12)
ax.set_ylabel('Catalogue Coverage (%)', fontsize=12)
ax.set_title('Experiment B: Catalogue Coverage vs. Rating Density\n'
             '(Fixed 62,423-film catalogue)', fontsize=13)
ax.legend(fontsize=10)
ax.grid(True, linestyle='--', alpha=0.4)
plt.tight_layout()

cov_fig_B = OUTPUT_DIR + 'sparsity_expB_coverage_curves.png'
plt.savefig(cov_fig_B, dpi=150, bbox_inches='tight')
plt.close()
print(f'Saved  →  {cov_fig_B}')

# mann-Whitney U test at Experiment B crossover
if crossover_sparsity_B is not None:
    cb_scores_B  = per_user_ndcg_B.get((crossover_sparsity_B, 'content_based'),  [])
    pen_scores_B = per_user_ndcg_B.get((crossover_sparsity_B, 'penalized_svd'),  [])

    if len(cb_scores_B) > 0 and len(pen_scores_B) > 0:
        u_stat_B, p_value_B = stats.mannwhitneyu(
            cb_scores_B, pen_scores_B, alternative='greater'
        )
        significant_B = p_value_B < 0.05

        print('\nMann-Whitney U (Exp B crossover):')
        print(f'  U statistic             : {u_stat_B:,.0f}')
        print(f'  P-value                 : {p_value_B:.4e}')
        print(f'  Significant (α=0.05)    : {significant_B}')
        print(f'  Content-based mean NDCG : {np.mean(cb_scores_B):.4f}')
        print(f'  Penalised SVD mean NDCG : {np.mean(pen_scores_B):.4f}')

        mw_out_B = OUTPUT_DIR + 'sparsity_expB_crossover_mannwhitney.csv'
        pd.DataFrame([{
            'experiment'                  : 'B_fixed_catalogue',
            'crossover_sparsity_level'    : crossover_sparsity_B,
            'approx_total_ratings'        : int(len(full_ratings) * crossover_sparsity_B),
            'avg_ratings_per_movie'       : avg_at_crossover,
            'u_statistic'                 : u_stat_B,
            'p_value'                     : p_value_B,
            'significant_at_0.05'         : significant_B,
            'content_based_mean_ndcg10'   : np.mean(cb_scores_B),
            'penalized_svd_mean_ndcg10'   : np.mean(pen_scores_B),
            'n_users_content_based'       : len(cb_scores_B),
            'n_users_penalized_svd'       : len(pen_scores_B),
        }]).to_csv(mw_out_B, index=False)
        print(f'Saved  →  {mw_out_B}')
    else:
        print('Insufficient per-user scores for Mann-Whitney test.')
else:
    print('No crossover identified in Experiment B — Mann-Whitney test skipped.')

# side-by-side plot: Exp A vs B

exp_a_path = OUTPUT_DIR + 'sparsity_results.csv'
if os.path.exists(exp_a_path):
    results_df_A = pd.read_csv(exp_a_path)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=True)

    for ax, (df, title, exp_label) in zip(
        axes,
        [
            (results_df_A, 'Experiment A\n(Shrinking Catalogue)', 'A'),
            (results_df_B, 'Experiment B\n(Fixed Catalogue: 62,423 films)', 'B'),
        ]
    ):
        for model in MODEL_NAMES:
            sub = df[df['model'] == model].sort_values('sparsity_level')
            if sub.empty:
                continue

            x    = sub['sparsity_level'].values
            y    = sub['ndcg10'].values

            # CI columns may be named differently; handle gracefully
            y_lo_col = 'ndcg10_ci_lower' if 'ndcg10_ci_lower' in sub.columns else None
            y_hi_col = 'ndcg10_ci_upper' if 'ndcg10_ci_upper' in sub.columns else None

            ax.plot(x, y, color=COLOURS[model], marker=MARKERS[model],
                    label=LABELS[model], linewidth=2, markersize=7)

            if y_lo_col and y_hi_col:
                ax.fill_between(x,
                                sub[y_lo_col].values,
                                sub[y_hi_col].values,
                                color=COLOURS[model], alpha=0.12)

        # crossover lines
        if exp_label == 'A':
            # load crossover from saved CSV if available
            mw_a_path = OUTPUT_DIR + 'sparsity_expA_crossover_mannwhitney.csv'
            if os.path.exists(mw_a_path):
                cs_a = pd.read_csv(mw_a_path)['crossover_sparsity_level'].iloc[0]
                ax.axvline(cs_a, color='black', linestyle='--', linewidth=1.3,
                           label=f'Crossover @ {cs_a}')
        else:
            if crossover_sparsity_B is not None:
                ax.axvline(crossover_sparsity_B, color='black', linestyle='--',
                           linewidth=1.3,
                           label=f'Crossover @ {crossover_sparsity_B}')

        ax.set_xscale('log')
        ax.set_xlabel('Sparsity Level', fontsize=11)
        ax.set_ylabel('NDCG@10', fontsize=11)
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.4)

    fig.suptitle('NDCG@10 vs. Sparsity: Shrinking vs. Fixed Catalogue\n'
                 '(shaded = 95% bootstrap CI, n=500 users)', fontsize=13)
    plt.tight_layout()

    side_by_side_path = OUTPUT_DIR + 'sparsity_expAB_side_by_side.png'
    plt.savefig(side_by_side_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved  →  {side_by_side_path}')
else:
    print(f'Experiment A results not found at {exp_a_path} — side-by-side plot skipped.')

print('\n✓ run_05b.py complete.')
