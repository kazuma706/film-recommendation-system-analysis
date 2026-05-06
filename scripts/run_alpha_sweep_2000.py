"""
Re-run hybrid alpha sweep with 2000 eval users (was 500).
"""

import random
import subprocess

import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

DATA_DIR   = '/Users/kazuma/Desktop/film data analysis/ml-25m/'
OUTPUT_DIR = '/Users/kazuma/Desktop/film data analysis/popularity-bias-project/outputs/'

RANDOM_SEED  = 42
N_EVAL_USERS = 2_000
N_BOOTSTRAP  = 1_000
TOP_N        = 10
N_SAMPLE     = 2_000_000
ALPHA_VALUES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


# helpers

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


if __name__ == '__main__':
    # load models
    print('Loading models...')
    svd_model = joblib.load(OUTPUT_DIR + 'svd_model.pkl')
    cb_art    = joblib.load(OUTPUT_DIR + 'content_based_model.pkl')
    tfidf_matrix  = cb_art['tfidf_matrix']
    movie_ids_cb  = cb_art['movie_ids']
    mid_to_idx_cb = cb_art['movie_id_to_idx']

    # load 2M sample
    print('Loading 2M sample...')
    ratings = pd.read_csv(DATA_DIR + 'ratings.csv')
    sample  = ratings.sample(n=N_SAMPLE, random_state=RANDOM_SEED)
    del ratings

    # long-tail definition
    print('Computing tail IDs from full dataset...')
    ratings_full = pd.read_csv(DATA_DIR + 'ratings.csv', usecols=['movieId'])
    film_pop = (
        ratings_full.groupby('movieId').size()
        .reset_index(name='cnt')
        .sort_values('cnt', ascending=False)
        .reset_index(drop=True)
    )
    film_pop['cumpct'] = film_pop['cnt'].cumsum() / film_pop['cnt'].sum() * 100
    cutoff   = (film_pop['cumpct'] >= 80).idxmax()
    tail_ids = set(film_pop.loc[cutoff + 1:, 'movieId'])
    all_movie_ids = set(film_pop['movieId'])
    print(f'  All films: {len(all_movie_ids):,}  |  Tail: {len(tail_ids):,}')
    del ratings_full, film_pop

    # train / test split
    train_df, test_df = train_test_split(sample, test_size=0.2, random_state=RANDOM_SEED)
    print(f'  Train: {len(train_df):,}  Test: {len(test_df):,}')

    # SVD index structures
    ts = svd_model.trainset
    inner_to_raw     = np.array([int(ts.to_raw_iid(i)) for i in range(ts.n_items)], dtype=np.int64)
    raw_uid_to_inner = {int(ts.to_raw_uid(u)): u for u in range(ts.n_users)}

    # 2000 eval users (intersection of train, test, SVD trainset)
    eligible = (
        set(train_df['userId'].unique())
        & set(test_df['userId'].unique())
        & set(raw_uid_to_inner.keys())
    )
    random.seed(RANDOM_SEED)
    eval_sample = random.sample(sorted(eligible), min(N_EVAL_USERS, len(eligible)))
    print(f'  Eval users: {len(eval_sample):,}  (requested {N_EVAL_USERS})')

    # precompute lookups
    print('Building lookups...')
    test_lookup = (
        test_df.groupby('userId', group_keys=False)
        .apply(lambda x: dict(zip(x['movieId'].astype(int), x['rating'])))
        .to_dict()
    )
    train_seen_lookup = train_df.groupby('userId')['movieId'].apply(set).to_dict()
    train_ratings_lookup = (
        train_df.groupby('userId', group_keys=False)
        .apply(lambda x: dict(zip(x['movieId'].astype(int), x['rating'])))
        .to_dict()
    )

    # per-user SVD scores and CB profiles (compute once, reuse across α)
    print('Pre-computing SVD scores and CB profiles for eval users...')
    user_svd_norm   = {}   # uid -> {movie_id: norm_score}
    user_cb_scores  = {}   # uid -> {movie_id: cb_score}
    user_seen       = {}   # uid -> set of seen movie_ids

    for i, uid in enumerate(eval_sample):
        if i % 200 == 0:
            print(f'  ... {i}/{len(eval_sample)}', flush=True)

        inner_uid = raw_uid_to_inner[uid]
        seen      = train_seen_lookup.get(uid, set())
        user_seen[uid] = seen

        # SVD batch scores
        svd_sc = (
            ts.global_mean
            + svd_model.bu[inner_uid]
            + svd_model.bi
            + svd_model.qi.dot(svd_model.pu[inner_uid])
        )
        user_svd_norm[uid] = {
            int(inner_to_raw[i]): (float(svd_sc[i]) - 0.5) / 4.5
            for i in range(len(inner_to_raw))
            if int(inner_to_raw[i]) not in seen
        }

        # CB profile
        cb_idx, cb_wts = [], []
        for i_inn, r in ts.ur[inner_uid]:
            mid = int(ts.to_raw_iid(i_inn))
            if mid in mid_to_idx_cb:
                cb_idx.append(mid_to_idx_cb[mid])
                cb_wts.append(r / 5.0)

        if cb_idx:
            w       = np.array(cb_wts, dtype=np.float32)
            profile = np.asarray(tfidf_matrix[cb_idx].T.dot(w)).ravel()
            norm    = np.linalg.norm(profile)
            if norm > 0:
                profile /= norm
                sims = tfidf_matrix.dot(profile)
                user_cb_scores[uid] = {
                    int(movie_ids_cb[j]): float(sims[j])
                    for j in range(len(movie_ids_cb))
                    if int(movie_ids_cb[j]) not in seen
                }
            else:
                user_cb_scores[uid] = {}
        else:
            user_cb_scores[uid] = {}

    # alpha sweep
    print('\nRunning alpha sweep...')
    rows = []

    for alpha in ALPHA_VALUES:
        print(f'  α={alpha:.1f}', end='', flush=True)
        ndcg_vals   = []
        rec_film_set = set()

        for uid in eval_sample:
            relevant = test_lookup.get(uid, {})

            svd_d  = user_svd_norm.get(uid, {})
            cb_d   = user_cb_scores.get(uid, {})
            cands  = set(svd_d) | set(cb_d)

            hybrid_sc = {
                mid: (1 - alpha) * svd_d.get(mid, 0.0) + alpha * cb_d.get(mid, 0.0)
                for mid in cands
            }
            recs = sorted(hybrid_sc, key=hybrid_sc.get, reverse=True)[:TOP_N]

            rec_film_set.update(recs)
            if relevant:
                ndcg_vals.append(ndcg_at_k(recs, relevant))

        mean_ndcg    = float(np.mean(ndcg_vals)) if ndcg_vals else np.nan
        ci_lo, ci_hi = bootstrap_ci(ndcg_vals)
        cat_cov  = len(rec_film_set) / len(all_movie_ids) * 100
        lt_cov   = len(rec_film_set & tail_ids) / len(tail_ids) * 100

        rows.append(dict(
            alpha=alpha,
            ndcg10=round(mean_ndcg, 4),
            ci_lo=round(ci_lo, 4),
            ci_hi=round(ci_hi, 4),
            cat_cov=round(cat_cov, 4),
            lt_cov=round(lt_cov, 4),
        ))
        print(f'  NDCG={mean_ndcg:.4f} [{ci_lo:.4f}, {ci_hi:.4f}]  '
              f'cat={cat_cov:.2f}%  lt={lt_cov:.2f}%')

    # save CSVs
    df_new = pd.DataFrame(rows)

    # preserve old results before overwriting
    df_old = pd.read_csv(OUTPUT_DIR + 'hybrid_alpha_sweep.csv')
    df_old.to_csv(OUTPUT_DIR + 'hybrid_alpha_sweep_500users.csv', index=False)
    print(f'\nOld results backed up → hybrid_alpha_sweep_500users.csv')

    df_new.to_csv(OUTPUT_DIR + 'hybrid_alpha_sweep.csv', index=False)
    df_new.to_csv(OUTPUT_DIR + 'hybrid_alpha_sweep_2000users.csv', index=False)
    print(f'Saved → hybrid_alpha_sweep.csv')
    print(f'Saved → hybrid_alpha_sweep_2000users.csv')

    # print comparison table
    print('\n=== Alpha sweep results (2000 users) ===')
    print(df_new.to_string(index=False))

    print('\n=== CI width comparison (500 vs 2000 users) ===')
    df_old_cmp = pd.read_csv(OUTPUT_DIR + 'hybrid_alpha_sweep_500users.csv')
    print(f'{"α":>5}  {"old CI width":>12}  {"new CI width":>12}  {"narrowing":>10}')
    for _, r_new in df_new.iterrows():
        r_old = df_old_cmp[df_old_cmp['alpha'] == r_new['alpha']].iloc[0]
        old_w = r_old['ci_hi'] - r_old['ci_lo']
        new_w = r_new['ci_hi'] - r_new['ci_lo']
        print(f'{r_new["alpha"]:>5.1f}  {old_w:>12.4f}  {new_w:>12.4f}  {(old_w-new_w)/old_w*100:>9.1f}%')

    # plot hybrid_ndcg_vs_alpha.png
    print('\nPlotting...')
    fig, ax = plt.subplots(figsize=(8, 4.5))

    alphas  = df_new['alpha'].values
    ndcgs   = df_new['ndcg10'].values
    ci_los  = df_new['ci_lo'].values
    ci_his  = df_new['ci_hi'].values

    ax.fill_between(alphas, ci_los, ci_his, alpha=0.20, color='#2196F3', label='95% CI')
    ax.plot(alphas, ndcgs, color='#2196F3', linewidth=2.5, marker='o',
            markersize=7, markerfacecolor='white', markeredgewidth=2, label='NDCG@10')

    # mark optimal alpha
    best_idx   = int(np.argmax(ndcgs))
    best_alpha = alphas[best_idx]
    best_ndcg  = ndcgs[best_idx]
    ax.axvline(best_alpha, color='black', linestyle='--', linewidth=1.2, alpha=0.6)
    ax.annotate(
        f'α={best_alpha:.1f}\nNDCG={best_ndcg:.4f}',
        xy=(best_alpha, best_ndcg),
        xytext=(best_alpha + 0.07, best_ndcg - 0.003),
        fontsize=9,
        arrowprops=dict(arrowstyle='->', color='black', lw=1.2),
    )

    ax.set_xlabel('Content-Based Weight α', fontsize=12)
    ax.set_ylabel('NDCG@10', fontsize=12)
    ax.set_title(
        f'Hybrid Model: NDCG@10 vs α  (n={N_EVAL_USERS:,} eval users, '
        f'1,000 bootstrap resamples)',
        fontsize=12,
    )
    ax.set_xticks(ALPHA_VALUES)
    ax.set_xlim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    plt.tight_layout()

    png_path = OUTPUT_DIR + 'hybrid_ndcg_vs_alpha.png'
    fig.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved → {png_path}')

    # regenerate HTML dashboard with updated alpha sweep
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
