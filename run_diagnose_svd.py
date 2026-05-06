"""
Diagnostic script — checks why SVD batch scoring gave NDCG@10 = 0.
"""

import numpy as np
import pandas as pd
import random
import joblib
import os
from scipy import stats

from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split as surprise_split

DATA_DIR   = '../ml-25m/'
OUTPUT_DIR = 'outputs/'
RANDOM_SEED = 42
N_EVAL_USERS = 500
TOP_N = 10

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# 1. Load data
print('Loading ratings...')
full_ratings = pd.read_csv(DATA_DIR + 'ratings.csv')
print(f'  Full dataset: {len(full_ratings):,} ratings, '
      f'{full_ratings["movieId"].nunique():,} movies, '
      f'{full_ratings["userId"].nunique():,} users')

subset = full_ratings.sample(n=2_000_000, random_state=RANDOM_SEED)
print(f'  Subsample   : {len(subset):,} ratings, '
      f'{subset["movieId"].nunique():,} movies')

print('Loading CB artefact for full catalogue...')
cb_art = joblib.load(OUTPUT_DIR + 'content_based_model.pkl')
all_catalogue_mids = set(int(m) for m in cb_art['movie_ids'])
FULL_CATALOGUE_SIZE = len(all_catalogue_mids)
print(f'  Full catalogue: {FULL_CATALOGUE_SIZE:,} films')

# 2. Build Surprise dataset & train SVD
print('\nBuilding Surprise dataset and splitting 80/20...')
reader   = Reader(rating_scale=(0.5, 5.0))
sur_data = Dataset.load_from_df(subset[['userId', 'movieId', 'rating']], reader)
trainset, testset = surprise_split(sur_data, test_size=0.2, random_state=RANDOM_SEED)

print(f'  Train: {trainset.n_ratings:,} ratings, '
      f'{trainset.n_users:,} users, {trainset.n_items:,} items')

print('Training SVD (k=100)...')
svd = SVD(n_factors=100, random_state=RANDOM_SEED)
svd.fit(trainset)
preds = svd.test(testset)
rmse  = accuracy.rmse(preds, verbose=False)
print(f'  RMSE = {rmse:.4f}')

# helper: build inner→raw arrays
ts = svd.trainset

inner_to_raw_uid = np.array([int(ts.to_raw_uid(u)) for u in range(ts.n_users)], dtype=np.int64)
inner_to_raw_iid = np.array([int(ts.to_raw_iid(i)) for i in range(ts.n_items)],  dtype=np.int64)

# raw → inner lookup (fast dict)
raw_to_inner_uid = {int(ts.to_raw_uid(u)): u for u in range(ts.n_users)}
raw_to_inner_iid = {int(ts.to_raw_iid(i)): i for i in range(ts.n_items)}

print(f'\n  trainset.n_items = {ts.n_items}')
print(f'  inner_to_raw_iid shape: {inner_to_raw_iid.shape}')
print(f'  svd.bi shape: {svd.bi.shape}')
print(f'  svd.qi shape: {svd.qi.shape}')

# 3. Build evaluation sample
trainset_df = pd.DataFrame(
    [(int(ts.to_raw_uid(u)), int(ts.to_raw_iid(i)), float(r))
     for u, i, r in ts.all_ratings()],
    columns=['userId', 'movieId', 'rating']
)
testset_df = pd.DataFrame(
    [(int(uid), int(iid), float(r)) for uid, iid, r in testset],
    columns=['userId', 'movieId', 'rating']
)

train_users = set(trainset_df['userId'].unique())
test_users  = set(testset_df['userId'].unique())
eval_users  = list(train_users & test_users)
random.seed(RANDOM_SEED)
eval_sample = random.sample(eval_users, min(N_EVAL_USERS, len(eval_users)))
print(f'\n  Evaluation users: {len(eval_sample)}')

test_lookup  = testset_df.groupby('userId').apply(
    lambda x: dict(zip(x['movieId'], x['rating'])), include_groups=False
).to_dict()
train_lookup = trainset_df.groupby('userId')['movieId'].apply(set).to_dict()

# 4. Batch scoring helper
def batch_scores_for_user(raw_uid):
    """Return score array indexed by inner item index, or None if user unknown."""
    if raw_uid not in raw_to_inner_uid:
        return None
    inner_uid = raw_to_inner_uid[raw_uid]
    return (ts.global_mean
            + svd.bu[inner_uid]
            + svd.bi
            + svd.qi.dot(svd.pu[inner_uid]))   # shape: (n_items,)

# diagnostic A: Score comparison on 1000 random pairs
print('\n' + '='*70)
print('DIAGNOSTIC A: per-pair predict() vs batch scores on 1000 random pairs')
print('='*70)

# sample 1000 (user, item) pairs that exist in the training set
all_train_pairs = [(int(ts.to_raw_uid(u)), int(ts.to_raw_iid(i)))
                   for u, i, _ in ts.all_ratings()]
random.seed(RANDOM_SEED)
sample_pairs = random.sample(all_train_pairs, min(1000, len(all_train_pairs)))

predict_scores = []
batch_scores   = []

for raw_uid, raw_iid in sample_pairs:
    # per-pair via predict()
    p = svd.predict(str(raw_uid), str(raw_iid))
    predict_scores.append(p.est)

    # batch
    sc = batch_scores_for_user(raw_uid)
    if sc is not None and raw_iid in raw_to_inner_iid:
        batch_scores.append(float(sc[raw_to_inner_iid[raw_iid]]))
    else:
        batch_scores.append(np.nan)

predict_scores = np.array(predict_scores)
batch_scores   = np.array(batch_scores)
valid = ~np.isnan(batch_scores)

print(f'  Pairs with valid batch score : {valid.sum()} / {len(sample_pairs)}')
if valid.sum() > 0:
    corr, _ = stats.pearsonr(predict_scores[valid], batch_scores[valid])
    mae_diff = np.mean(np.abs(predict_scores[valid] - batch_scores[valid]))
    print(f'  Pearson r (predict vs batch) : {corr:.6f}')
    print(f'  Mean |predict - batch|        : {mae_diff:.6f}')
    print(f'  Max  |predict - batch|        : {np.max(np.abs(predict_scores[valid] - batch_scores[valid])):.6f}')
    print(f'  predict() range              : [{predict_scores[valid].min():.4f}, {predict_scores[valid].max():.4f}]')
    print(f'  batch    range               : [{batch_scores[valid].min():.4f},  {batch_scores[valid].max():.4f}]')

# diagnostic B: Predict() call with int vs str raw IDs
print('\n' + '='*70)
print('DIAGNOSTIC B: predict() with int IDs vs str IDs')
print('='*70)

raw_uid_ex, raw_iid_ex = sample_pairs[0]
p_str = svd.predict(str(raw_uid_ex), str(raw_iid_ex))
p_int = svd.predict(raw_uid_ex, raw_iid_ex)        # raw int
p_unk = svd.predict(str(raw_uid_ex), str(raw_iid_ex + 999999))  # unknown item

print(f'  predict(str_uid, str_iid)  → est={p_str.est:.4f}  details={p_str.details}')
print(f'  predict(int_uid, int_iid)  → est={p_int.est:.4f}  details={p_int.details}')
print(f'  predict(str_uid, UNKNOWN)  → est={p_unk.est:.4f}  details={p_unk.details}')

# diagnostic C: Ranking personalisation (Jaccard across 50 users)
print('\n' + '='*70)
print('DIAGNOSTIC C: ranking personalisation — Jaccard@10 across 50 user pairs')
print('='*70)

sample50 = eval_sample[:50]
top10_batch   = {}
top10_predict = {}

for uid in sample50:
    seen = train_lookup.get(uid, set())

    # batch top-10
    sc = batch_scores_for_user(uid)
    if sc is not None:
        ranked = sorted(
            [(int(inner_to_raw_iid[i]), float(sc[i]))
             for i in range(ts.n_items) if int(inner_to_raw_iid[i]) not in seen],
            key=lambda x: x[1], reverse=True
        )
        top10_batch[uid] = set(mid for mid, _ in ranked[:TOP_N])
    else:
        top10_batch[uid] = set()

    # predict() top-10 (from training movies only, for speed)
    cands = [int(inner_to_raw_iid[i]) for i in range(ts.n_items)
             if int(inner_to_raw_iid[i]) not in seen]
    # subsample candidates for speed (predict() is slow over 50K items)
    random.seed(RANDOM_SEED)
    cand_sample = random.sample(cands, min(5000, len(cands)))
    preds_u = [(mid, svd.predict(str(uid), str(mid)).est) for mid in cand_sample]
    preds_u.sort(key=lambda x: x[1], reverse=True)
    top10_predict[uid] = set(mid for mid, _ in preds_u[:TOP_N])

# pairwise Jaccard
jaccards_batch   = []
jaccards_predict = []
users_list = list(sample50)
for i in range(len(users_list)):
    for j in range(i+1, len(users_list)):
        u, v = users_list[i], users_list[j]
        b_u, b_v = top10_batch[u],   top10_batch[v]
        p_u, p_v = top10_predict[u], top10_predict[v]
        if b_u | b_v:
            jaccards_batch.append(len(b_u & b_v) / len(b_u | b_v))
        if p_u | p_v:
            jaccards_predict.append(len(p_u & p_v) / len(p_u | p_v))

print(f'  Mean pairwise Jaccard@10 (batch)   : {np.mean(jaccards_batch):.4f}  '
      f'(1.0 = identical for all users, ~0 = fully personalised)')
print(f'  Mean pairwise Jaccard@10 (predict) : {np.mean(jaccards_predict):.4f}')

# show top-10 for first 3 users (batch)
print('\n  Top-10 movie IDs per user (batch scoring, first 3 eval users):')
for uid in users_list[:3]:
    print(f'    user {uid}: {sorted(top10_batch[uid])}')

# diagnostic D: Variance decomposition (dot-product vs bias terms)
print('\n' + '='*70)
print('DIAGNOSTIC D: variance of score components across 20 users × all items')
print('='*70)

sample20 = eval_sample[:20]
all_dot_products = []
all_bi           = []
all_bu_plus_mu   = []

for uid in sample20:
    if uid not in raw_to_inner_uid:
        continue
    inner_uid = raw_to_inner_uid[uid]
    dot = svd.qi.dot(svd.pu[inner_uid])          # shape (n_items,)
    all_dot_products.append(dot)
    all_bi.append(svd.bi)
    all_bu_plus_mu.append(np.full(ts.n_items, ts.global_mean + svd.bu[inner_uid]))

dot_arr    = np.concatenate(all_dot_products)
bi_arr     = np.concatenate(all_bi)
const_arr  = np.concatenate(all_bu_plus_mu)

print(f'  global_mean               : {ts.global_mean:.4f}')
print(f'  bu range (all users)      : [{svd.bu.min():.4f}, {svd.bu.max():.4f}]')
print(f'  bi range (all items)      : [{svd.bi.min():.4f}, {svd.bi.max():.4f}]')
print(f'  q_i @ p_u range           : [{dot_arr.min():.4f}, {dot_arr.max():.4f}]')
print()
print(f'  Variance of bi            : {np.var(bi_arr):.6f}')
print(f'  Variance of q_i @ p_u    : {np.var(dot_arr):.6f}')
print(f'  Var(dot) / Var(bi)        : {np.var(dot_arr)/np.var(bi_arr):.4f}')
print()

# check if dot product term meaningfully differentiates rankings
# i.e., does the rank correlation between full score and bi-only differ across users?
print('  Rank-correlation of (full score) vs (bi only), for 5 users:')
for uid in sample20[:5]:
    if uid not in raw_to_inner_uid:
        continue
    inner_uid = raw_to_inner_uid[uid]
    full_sc  = ts.global_mean + svd.bu[inner_uid] + svd.bi + svd.qi.dot(svd.pu[inner_uid])
    bi_only  = svd.bi
    rho, _   = stats.spearmanr(full_sc, bi_only)
    print(f'    user {uid}  spearman(full, bi_only) = {rho:.4f}')

# diagnostic E: Items in test set — are they in training?
print('\n' + '='*70)
print('DIAGNOSTIC E: test set items vs training item set')
print('='*70)

test_mids_all   = set(testset_df['movieId'].unique())
train_mids_all  = set(inner_to_raw_iid.tolist())
test_only_mids  = test_mids_all - train_mids_all
both_mids       = test_mids_all & train_mids_all

print(f'  Unique items in test set           : {len(test_mids_all):,}')
print(f'  Items in test AND training set     : {len(both_mids):,}')
print(f'  Items in test ONLY (not in train)  : {len(test_only_mids):,}')
print(f'  → SVD can only recommend training items; '
      f'{len(test_only_mids)/len(test_mids_all)*100:.1f}% of test items are unreachable')

# per-user: what fraction of each eval user's test items are in the training set?
per_user_reachable = []
for uid in eval_sample:
    relevant = test_lookup.get(uid, {})
    if not relevant:
        continue
    reachable = sum(1 for mid in relevant if mid in train_mids_all)
    per_user_reachable.append(reachable / len(relevant))

print(f'\n  Per-user fraction of test items reachable by SVD:')
print(f'    Mean  : {np.mean(per_user_reachable):.4f}')
print(f'    Median: {np.median(per_user_reachable):.4f}')
print(f'    Min   : {np.min(per_user_reachable):.4f}')
print(f'    Users with 0 reachable test items: '
      f'{sum(1 for x in per_user_reachable if x == 0)}')

# diagnostic F: NDCG@10 evaluation — both methods, same users
print('\n' + '='*70)
print('DIAGNOSTIC F: NDCG@10 — batch scoring vs predict() on same 500 users')
print('='*70)

def ndcg_at_k(recommended, relevant_ratings, k=10):
    if not relevant_ratings:
        return 0.0
    dcg = sum(
        (relevant_ratings[item] / 5.0) / np.log2(rank + 2)
        for rank, item in enumerate(recommended[:k])
        if item in relevant_ratings
    )
    ideal = sorted(relevant_ratings.values(), reverse=True)[:k]
    idcg  = sum((r/5.0) / np.log2(rank+2) for rank, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0

ndcg_batch   = []
ndcg_predict = []
ndcg_batch_train_only   = []  # batch but only candidates from training set
ndcg_predict_train_only = []

skipped = 0
for uid in eval_sample:
    relevant = test_lookup.get(uid, {})
    if not relevant:
        skipped += 1
        continue
    seen = train_lookup.get(uid, set())

    # method 1: Batch scoring, candidates = all training items
    sc = batch_scores_for_user(uid)
    if sc is not None:
        ranked = sorted(
            [(int(inner_to_raw_iid[i]), float(sc[i]))
             for i in range(ts.n_items) if int(inner_to_raw_iid[i]) not in seen],
            key=lambda x: x[1], reverse=True
        )
        recs_batch = [mid for mid, _ in ranked[:TOP_N]]
        ndcg_batch.append(ndcg_at_k(recs_batch, relevant))
        ndcg_batch_train_only.append(ndcg_at_k(recs_batch, relevant))

    # method 2: predict() on all training items
    cands = [int(inner_to_raw_iid[i]) for i in range(ts.n_items)
             if int(inner_to_raw_iid[i]) not in seen]
    preds_u = [(mid, svd.predict(str(uid), str(mid)).est) for mid in cands]
    preds_u.sort(key=lambda x: x[1], reverse=True)
    recs_predict = [mid for mid, _ in preds_u[:TOP_N]]
    ndcg_predict.append(ndcg_at_k(recs_predict, relevant))
    ndcg_predict_train_only.append(ndcg_at_k(recs_predict, relevant))

print(f'  Evaluated users : {len(ndcg_batch)} (skipped {skipped} with no test items)')
print()
print(f'  NDCG@10 — Batch scoring (training candidates only):')
print(f'    Mean  = {np.mean(ndcg_batch):.4f}')
print(f'    >0    = {sum(1 for x in ndcg_batch if x > 0)} / {len(ndcg_batch)} users')
print()
print(f'  NDCG@10 — predict() (training candidates only):')
print(f'    Mean  = {np.mean(ndcg_predict):.4f}')
print(f'    >0    = {sum(1 for x in ndcg_predict if x > 0)} / {len(ndcg_predict)} users')

# diagnostic G: Are the test_lookup keys int or something else?
print('\n' + '='*70)
print('DIAGNOSTIC G: key type audit — test_lookup, train_lookup, recs')
print('='*70)

uid_ex = eval_sample[0]
rel_ex = test_lookup.get(uid_ex, {})
seen_ex = train_lookup.get(uid_ex, set())

sc_ex = batch_scores_for_user(uid_ex)
ranked_ex = sorted(
    [(int(inner_to_raw_iid[i]), float(sc_ex[i]))
     for i in range(ts.n_items) if int(inner_to_raw_iid[i]) not in seen_ex],
    key=lambda x: x[1], reverse=True
)
top10_ex = [mid for mid, _ in ranked_ex[:TOP_N]]

print(f'  eval_sample[0]  = {uid_ex}  type={type(uid_ex).__name__}')
print(f'  test_lookup key type  : {type(next(iter(test_lookup.keys()))).__name__}')
print(f'  test_lookup[uid] = {rel_ex}')
print(f'  type of test_lookup[uid] keys: {type(next(iter(rel_ex.keys()))).__name__ if rel_ex else "N/A"}')
print(f'  train_lookup[uid] (first 5): {list(seen_ex)[:5]}')
print(f'  type of train_lookup values: {type(next(iter(seen_ex))).__name__ if seen_ex else "N/A"}')
print(f'  top-10 batch recs (first 5): {top10_ex[:5]}')
print(f'  type of batch recs: {type(top10_ex[0]).__name__ if top10_ex else "N/A"}')
print()
# direct intersection check
print(f'  Intersection of top10_batch and relevant (direct): '
      f'{set(top10_ex) & set(rel_ex.keys())}')

# explicit type-coerced check
top10_int = [int(m) for m in top10_ex]
rel_int   = {int(k): v for k, v in rel_ex.items()}
print(f'  After explicit int() coercion: {set(top10_int) & set(rel_int.keys())}')

# diagnostic H: Does predict() use str IDs internally?
print('\n' + '='*70)
print('DIAGNOSTIC H: Surprise internal ID format check')
print('='*70)
# sample raw IID from the trainset
raw_iid_sample = ts.to_raw_iid(0)
raw_uid_sample = ts.to_raw_uid(0)
print(f'  ts.to_raw_iid(0) = {repr(raw_iid_sample)}  type={type(raw_iid_sample).__name__}')
print(f'  ts.to_raw_uid(0) = {repr(raw_uid_sample)}  type={type(raw_uid_sample).__name__}')
print()
# does int(ts.to_raw_iid(0)) == the movieId we stored?
raw_iid_0_int = int(raw_iid_sample)
print(f'  int(ts.to_raw_iid(0)) = {raw_iid_0_int}')

# check if the raw IIDs in the trainset are the exact same integer value as the
# movieIds in the original subset DataFrame
sample_mids = subset['movieId'].sample(5, random_state=42).tolist()
print(f'\n  Checking trainset raw IIDs for subset movieIds: {sample_mids}')
for mid in sample_mids:
    try:
        inner = ts.to_inner_iid(str(mid))
        back  = ts.to_raw_iid(inner)
        print(f'    movieId {mid} → inner {inner} → back to raw {repr(back)} '
              f'→ int(back)={int(back)} match={int(back)==mid}')
    except ValueError:
        print(f'    movieId {mid} → NOT in trainset')

# summary
print('\n' + '='*70)
print('SUMMARY')
print('='*70)
print(f'  Batch NDCG@10 mean   : {np.mean(ndcg_batch):.4f}')
print(f'  predict() NDCG@10    : {np.mean(ndcg_predict):.4f}')
print(f'  Score correlation    : {corr:.6f}')
print(f'  Mean Jaccard (batch) : {np.mean(jaccards_batch):.4f}')
print(f'  Test-only items      : {len(test_only_mids):,} ({len(test_only_mids)/len(test_mids_all)*100:.1f}%)')
print(f'  Var(dot)/Var(bi)     : {np.var(dot_arr)/np.var(bi_arr):.4f}')
