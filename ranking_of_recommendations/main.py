import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit
import lightgbm as lgb
from tqdm import tqdm

# 1. Загрузка данных
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("Train columns:", train.columns.tolist())
print("Test columns:", test.columns.tolist())

# 2. Подготовка признаков
category_dummies = pd.get_dummies(train['category'], prefix='cat', dtype=int)
train = pd.concat([train, category_dummies], axis=1)

category_dummies_test = pd.get_dummies(test['category'], prefix='cat', dtype=int)
for col in category_dummies.columns:
    if col not in category_dummies_test.columns:
        category_dummies_test[col] = 0
test = pd.concat([test, category_dummies_test], axis=1)

feature_cols = [f'feat_{i}' for i in range(1, 9)] + list(category_dummies.columns)
X = train[feature_cols]
y = train['relevance']
X_test = test[feature_cols]

# 3. Разделение по запросам (валидационная выборка – 20% запросов)
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(gss.split(X, groups=train['query_id']))

X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]
query_val = train['query_id'].iloc[val_idx].values

print(f"Train queries: {len(train['query_id'].iloc[train_idx].unique())}")
print(f"Val queries: {len(np.unique(query_val))}")

# 4. Обучение модели регрессии (LightGBM)
params = {
    'objective': 'regression',
    'metric': 'mae',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'random_state': 42
}
model = lgb.LGBMRegressor(**params, n_estimators=1000)
model.fit(X_train, y_train,
          eval_set=[(X_val, y_val)],
          eval_metric='mae',
          callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)])

# 5. Функция вычисления DA-NDCG
def da_ndcg(relevance, categories):
    """
    relevance: list of true relevance (0-4) в порядке ранжирования
    categories: list of categories (int) в том же порядке
    """
    dcg = 0.0
    prev_cat = None
    for i, (rel, cat) in enumerate(zip(relevance, categories)):
        mult = 0.1 if (prev_cat is not None and cat == prev_cat) else 1.0
        dcg += (rel * mult) / np.log2(i + 2)
        prev_cat = cat
    # IDCG: сортируем relevance по убыванию, без штрафов
    ideal_rel = sorted(relevance, reverse=True)
    ideal_dcg = 0.0
    for i, rel in enumerate(ideal_rel):
        ideal_dcg += rel / np.log2(i + 2)
    return dcg / ideal_dcg if ideal_dcg > 0 else 0.0

# 6. Жадный алгоритм переранжирования для одного запроса
def greedy_rerank(items_df):
    """
    items_df: DataFrame с колонками 'relevance_pred', 'category'
    Возвращает список item_id в порядке ранжирования
    """
    items = items_df.copy()
    items = items.sort_values('relevance_pred', ascending=False).reset_index(drop=True)
    result = []
    last_cat = None
    remaining = items.index.tolist()
    for _ in range(len(items)):
        best_idx = None
        best_score = -1.0
        for idx in remaining:
            rel = items.loc[idx, 'relevance_pred']
            cat = items.loc[idx, 'category']
            effective = rel * (0.1 if (last_cat is not None and cat == last_cat) else 1.0)
            if best_idx is None or effective > best_score or (effective == best_score and rel > items.loc[best_idx, 'relevance_pred']):
                best_score = effective
                best_idx = idx
        result.append(items.loc[best_idx, 'item_id'])
        last_cat = items.loc[best_idx, 'category']
        remaining.remove(best_idx)
    return result

# 7. Оценка на валидации
val_data = train.iloc[val_idx].copy()
val_data['relevance_pred'] = model.predict(X_val)

val_baseline_scores = []
val_greedy_scores = []
for qid in tqdm(np.unique(query_val), desc="Validation"):
    q_mask = (val_data['query_id'] == qid)
    df_q = val_data[q_mask].copy()
    # Baseline: сортировка по предсказанной релевантности (без diversity)
    baseline_order = df_q.sort_values('relevance_pred', ascending=False)['item_id'].values
    baseline_rel = df_q.set_index('item_id').loc[baseline_order]['relevance'].values
    baseline_cat = df_q.set_index('item_id').loc[baseline_order]['category'].values
    baseline_ndcg = da_ndcg(baseline_rel, baseline_cat)
    val_baseline_scores.append(baseline_ndcg)
    # Greedy rerank
    greedy_order = greedy_rerank(df_q[['item_id', 'relevance_pred', 'category']])
    greedy_rel = df_q.set_index('item_id').loc[greedy_order]['relevance'].values
    greedy_cat = df_q.set_index('item_id').loc[greedy_order]['category'].values
    greedy_ndcg = da_ndcg(greedy_rel, greedy_cat)
    val_greedy_scores.append(greedy_ndcg)

print(f"Baseline DA-NDCG на валидации: {np.mean(val_baseline_scores):.4f}")
print(f"Greedy rerank DA-NDCG на валидации: {np.mean(val_greedy_scores):.4f}")

# 8. Финальное предсказание для теста
model_full = lgb.LGBMRegressor(**params, n_estimators=model.best_iteration_)
model_full.fit(X, y)

test['relevance_pred'] = model_full.predict(X_test)

submission_rows = []
for qid in tqdm(sorted(test['query_id'].unique()), desc="Test queries"):
    df_q = test[test['query_id'] == qid].copy()
    ordered_items = greedy_rerank(df_q[['item_id', 'relevance_pred', 'category']])
    for item in ordered_items:
        submission_rows.append([qid, item])

sub_df = pd.DataFrame(submission_rows, columns=['query_id', 'item_id'])
sub_df.to_csv('submission.csv', index=False)
print("submission.csv сохранён. Первые 20 строк:")
print(sub_df.head(20))