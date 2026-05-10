import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# 1. Анализ количества сезонов на магазин
season_counts = train.groupby('id')['season_month'].nunique()
print("Распределение числа сезонов на магазин в train:")
print(season_counts.value_counts().sort_index())

# 2. Подготовка признаков для прогноза сезона 3
# Создаём таблицу с target за сезоны 1,2,3
pivot_target = train.pivot_table(index='id', columns='season_month', values='target')
pivot_target.columns = [f'target_season_{c}' for c in pivot_target.columns]
pivot_target = pivot_target.reset_index()

# Добавляем средние значения других признаков за сезоны 1-2
other_feats = ['store_area', 'population_density', 'num_employees', 'marketing_spend',
               'foot_traffic', 'competitor_price', 'store_type', 'district_growth',
               'customer_loyalty', 'avg_transaction', 'promo_intensity', 'inventory_turnover']

for feat in other_feats:
    mean_12 = train[train['season_month'].isin([1,2])].groupby('id')[feat].mean()
    pivot_target[f'{feat}_mean_12'] = pivot_target['id'].map(mean_12)

# Удаляем строки, где нет target_season_3 (только те магазины, у которых есть все три сезона)
pivot_target = pivot_target.dropna(subset=['target_season_3'])

X = pivot_target.drop(['id', 'target_season_3'], axis=1)
y = pivot_target['target_season_3']

# Разделение на train/val (80/20) для оценки
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_val = rf.predict(X_val)
mae_val = mean_absolute_error(y_val, y_pred_val)
print(f"MAE на валидации (прогноз сезона 3 по сезонам 1-2): {mae_val:.2f}")

# 3. Подготовка тестовых признаков (для сезона 4)
# Для теста берём средние по сезонам 1-2 из train (по id). Для новых id – медиана.
test_features = test[['id']].copy()
for feat in other_feats:
    mean_12_dict = train[train['season_month'].isin([1,2])].groupby('id')[feat].mean().to_dict()
    test_features[f'{feat}_mean_12'] = test['id'].map(mean_12_dict)

# Заполняем пропуски (новые id) медианой по всем train
for col in test_features.columns:
    if col != 'id':
        test_features[col].fillna(test_features[col].median(), inplace=True)

X_test = test_features.drop('id', axis=1)
y_pred_test = rf.predict(X_test)

# 4. Сохранение результата
sub = pd.DataFrame({'id': test['id'], 'revenue': y_pred_test})
sub.to_csv('submission.csv', index=False)
print("submission.csv сохранён. Первые 5 строк:")
print(sub.head())