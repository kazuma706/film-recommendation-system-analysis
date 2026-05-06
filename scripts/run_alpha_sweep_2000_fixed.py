"""
Corrected hybrid alpha sweep with 2000 eval users.
Uses test set from SVD's own 80/20 split to avoid data leakage.
"""

import random
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

    ts = svd_model.trainset
    inner_to_raw     = np.array([int(ts.to_raw_iid(i)) for i in range(ts.n_items)], dtype=np.int64)
    raw_uid_to_inner = {int(ts.to_raw_uid(u)): u for u in range(ts.n_users)}
    print(f'  SVD trainset: {ts.n_users:,} users, {ts.n_items:,} items, {ts.n_ratings:,} ratings')

    # load 2M sample (exact same draw as original notebooks)
    print('Loading 2M sample...')
    ratings_full = pd.read_csv(DATA_DIR + 'ratings.csv')
    sample = ratings_full.sample(n=N_SAMPLE, random_state=RANDOM_SEED)
    del ratings_full
    print(f'  Sample: {len(sample):,} ratings')

    # long-tail definition (80/20 from full dataset)
    print('Computing tail IDs...')
    ratings_for_tail = pd.read_csv(DATA_DIR + 'ratings.csv', usecols=['movieId'])
    film_pop = (
        ratings_for_tail.groupby('movieId').size()
        .reset_index(name='cnt')
        .sort_values('cnt', ascending=False)
        .reset_index(drop=True)
    )
    film_pop['cumpct'] = film_pop['cnt'].cumsum() / film_pop['cnt'].sum() * 100
    cutoff    = (film_pop['cumpct'] >= 80).idxmax()
    tail_ids  = set(film_pop.loc[cutoff + 1:, 'movieId'])
    all_movie_ids = set(film_pop['movieId'])
    del ratings_for_tail, film_pop
    print(f'  All films: {len(all_movie_ids):,}  |  Tail: {len(tail_ids):,}')

    # rows in 2M sample not in SVD trainset = test set
    print('Reconstructing test set from SVD trainset membership...')
    train_uid_arr, train_iid_arr = [], []
    for u_inner in range(ts.n_users):
        uid = int(ts.to_raw_uid(u_inner))
        for i_inner, _ in ts.ur[u_inner]:
            train_uid_arr.append(uid)
            train_iid_arr.append(int(ts.to_raw_iid(i_inner)))

    train_pairs_df = pd.DataFrame({
        'userId':  train_uid_arr,
        'movieId': train_iid_arr,
        '_flag':   True,
    })
    sample_reset = sample.reset_index(drop=True)
    merged   = sample_reset.merge(train_pairs_df, on=['userId', 'movieId'], how='left')
    test_df  = sample_reset[merged['_flag'].isna().values].reset_index(drop=True)
    del train_uid_arr, train_iid_arr, train_pairs_df, merged, sample_reset
    print(f'  Test ratings: {len(test_df):,}  '
          f'(expected ~{int(N_SAMPLE * 0.2):,})')

    # sanity check: expected ~400K test ratings
    assert 350_000 < len(test_df) < 450_000, \
        f'Unexpected test size {len(test_df):,} — check sample/model alignment'

    # lookups
    print('Building lookups...')
    test_lookup = (
        test_df.groupby('userId', group_keys=False)
        .apply(lambda x: dict(zip(x['movieId'].astype(int), x['rating'])))
        .to_dict()
    )
    # train_seen derived from SVD trainset — same split as above
    train_seen_lookup = {
        int(ts.to_raw_uid(u)): {int(ts.to_raw_iid(i)) for i, _ in ts.ur[u]}
        for u in range(ts.n_users)
    }

    # 2000 eval users
    eligible = (
        set(raw_uid_to_inner.keys())
        & {uid for uid, rel in test_lookup.items() if rel}
    )
    random.seed(RANDOM_SEED)
    eval_sample = random.sample(sorted(eligible), min(N_EVAL_USERS, len(eligible)))
    print(f'  Eval users: {len(eval_sample):,}')

    # pre-compute SVD scores and CB profiles (once, reused across α)
    print('Pre-computing SVD scores and CB profiles...')
    user_svd_norm  = {}
    user_cb_scores = {}

    for i, uid in enumerate(eval_sample):
        if i % 200 == 0:
            print(f'  ... {i}/{len(eval_sample)}', flush=True)

        inner_uid = raw_uid_to_inner[uid]
        seen      = train_seen_lookup.get(uid, set())

        # SVD: batch score over all items, exclude seen
        svd_sc = (
            ts.global_mean
            + svd_model.bu[inner_uid]
            + svd_model.bi
            + svd_model.qi.dot(svd_model.pu[inner_uid])
        )
        user_svd_norm[uid] = {
            int(inner_to_raw[j]): (float(svd_sc[j]) - 0.5) / 4.5
            for j in range(len(inner_to_raw))
            if int(inner_to_raw[j]) not in seen
        }

        # CB: build profile from training ratings (ts.ur), exclude seen
        cb_idx, cb_wts = [], []
        for i_inn, r in ts.ur[inner_uid]:
            mid = int(ts.to_raw_iid(i_inn))
            if mid in mid_to_idx_cb:
                cb_idx.append(mid_to_idx_cb[mid])
                cb_wts.append(r / 5.0)

        if cb_idx:
            import numpy as np
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
        ndcg_vals    = []
        rec_film_set = set()

        for uid in eval_sample:
            relevant = test_lookup.get(uid, {})
            svd_d    = user_svd_norm.get(uid, {})
            cb_d     = user_cb_scores.get(uid, {})
            cands    = set(svd_d) | set(cb_d)

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
        print(f'  α={alpha:.1f}  NDCG={mean_ndcg:.4f} [{ci_lo:.4f}, {ci_hi:.4f}]  '
              f'cat={cat_cov:.2f}%  lt={lt_cov:.2f}%')

    df_new = pd.DataFrame(rows)

    # print comparison table
    df_500 = pd.read_csv(OUTPUT_DIR + 'hybrid_alpha_sweep_500users.csv')
    print('\n=== CI width comparison (500 vs 2000 users) ===')
    print(f'{"α":>5}  {"NDCG(500)":>10}  {"NDCG(2000)":>11}  '
          f'{"CI(500)":>10}  {"CI(2000)":>10}  {"narrowing":>10}')
    for _, r2 in df_new.iterrows():
        r5   = df_500[df_500['alpha'] == r2['alpha']].iloc[0]
        w5   = r5['ci_hi'] - r5['ci_lo']
        w2   = r2['ci_hi'] - r2['ci_lo']
        narr = (w5 - w2) / w5 * 100 if w5 > 0 else 0.0
        print(f'{r2["alpha"]:>5.1f}  {r5["ndcg10"]:>10.4f}  {r2["ndcg10"]:>11.4f}  '
              f'{w5:>10.4f}  {w2:>10.4f}  {narr:>9.1f}%')

    # save CSV
    df_new.to_csv(OUTPUT_DIR + 'hybrid_alpha_sweep_2000users.csv', index=False)
    print(f'\nSaved → hybrid_alpha_sweep_2000users.csv')

    # reference lines from hybrid_summary.csv and ease_expA_results.csv
    summary = pd.read_csv(OUTPUT_DIR + 'hybrid_summary.csv').set_index('model')
    ease_a  = pd.read_csv(OUTPUT_DIR + 'ease_expA_results.csv')
    ease_ref_ndcg = float(ease_a[ease_a['sparsity'] == 0.1]['ndcg'].values[0])  # 0.0346

    REF_MODELS = [
        ('Popularity Baseline', summary.loc['popularity_baseline', 'ndcg10'], '#636EFA', '--'),
        ('SVD',                 summary.loc['svd',                  'ndcg10'], '#EF553B', '--'),
        ('Penalised SVD',       summary.loc['penalized_svd',        'ndcg10'], '#00CC96', '--'),
        ('Content-Based',       summary.loc['content_based',        'ndcg10'], '#AB63FA', '--'),
        ('Hybrid RRF',          summary.loc['hybrid_rrf',           'ndcg10'], '#19D3F3', '--'),
        ('EASE (Exp A, s=0.1)', ease_ref_ndcg,                                '#00BCD4', '--'),
    ]

    # plot
    print('\nPlotting...')
    fig, ax = plt.subplots(figsize=(10, 5.5))

    alphas = df_new['alpha'].values
    ndcgs  = df_new['ndcg10'].values
    ci_los = df_new['ci_lo'].values
    ci_his = df_new['ci_hi'].values

    # reference lines (drawn first, behind main curve)
    for label, ndcg_val, colour, ls in REF_MODELS:
        ax.axhline(ndcg_val, color=colour, linestyle=ls, linewidth=1.2,
                   alpha=0.75, label=label)

    # main hybrid curve
    ax.fill_between(alphas, ci_los, ci_his, alpha=0.18, color='#2196F3', zorder=3)
    ax.plot(alphas, ndcgs, color='#2196F3', linewidth=2.5, marker='o',
            markersize=7, markerfacecolor='white', markeredgewidth=2,
            label='Hybrid NDCG@10', zorder=4)

    # best-alpha marker
    best_idx   = int(np.argmax(ndcgs))
    best_alpha = alphas[best_idx]
    best_ndcg  = ndcgs[best_idx]
    ax.axvline(best_alpha, color='#333333', linestyle=':', linewidth=1.4,
               alpha=0.7, zorder=2)
    ax.annotate(
        f'α={best_alpha:.1f}\nNDCG={best_ndcg:.4f}',
        xy=(best_alpha, best_ndcg),
        xytext=(best_alpha - 0.18, best_ndcg - 0.003),
        fontsize=9,
        arrowprops=dict(arrowstyle='->', color='#333333', lw=1.2),
        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='none', alpha=0.8),
    )

    ax.set_xlabel('Content-Based Weight α', fontsize=12)
    ax.set_ylabel('NDCG@10', fontsize=12)
    ax.set_title(
        f'Hybrid Model: NDCG@10 vs α  '
        f'(n={N_EVAL_USERS:,} eval users, {N_BOOTSTRAP:,} bootstrap resamples)',
        fontsize=12,
    )
    ax.set_xticks(ALPHA_VALUES)
    ax.set_xlim(-0.02, 1.02)
    ax.grid(True, alpha=0.3, zorder=0)
    ax.legend(fontsize=9, loc='upper left', ncol=2,
              framealpha=0.9, edgecolor='#cccccc')

    plt.tight_layout()
    png_path = OUTPUT_DIR + 'hybrid_ndcg_vs_alpha_2000users.png'
    fig.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved → {png_path}')
    print('\nAll done.')
