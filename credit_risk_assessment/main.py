import pandas as pd
import numpy as np
from sklearn.linear_model import QuantileRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold

# Загрузка данных
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Признаки (все кроме id и target)
features = [c for c in train.columns if c not in ['id', 'target']]

X = train[features].values
y = train['target'].values
X_test = test[features].values

# Кросс-валидация для оценки качества
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof = np.zeros(len(X))

for train_idx, val_idx in kf.split(X):
    model = QuantileRegressor(quantile=0.5, alpha=0.01, solver='highs')
    model.fit(X[train_idx], y[train_idx])
    oof[val_idx] = model.predict(X[val_idx])

mae = mean_absolute_error(y, oof)
print(f'OOF MAE: {mae:.2f}')
print(f'1 / (1 + MAE): {1 / (1 + mae):.6f}')

# Финальная модель на всех данных
final_model = QuantileRegressor(quantile=0.5, alpha=0.01, solver='highs')
final_model.fit(X, y)

# Предсказание
predictions = final_model.predict(X_test)

# Сохранение
submission = pd.DataFrame({'id': test['id'], 'revenue': predictions})
submission.to_csv('submission.csv', index=False)
print(f'submission.csv сохранён: {len(submission)} строк')