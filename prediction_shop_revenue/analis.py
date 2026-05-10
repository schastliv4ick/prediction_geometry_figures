import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Загрузка данных
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

features = [c for c in train.columns if c not in ['id', 'target']]
X = train[features]
y = train['target']

# Разделение на train/val для оценки
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_val)

# Ошибки
residuals = y_val - y_pred
mae = mean_absolute_error(y_val, y_pred)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
r2 = r2_score(y_val, y_pred)

print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"R2 : {r2:.4f}")

# 1. График: предсказанные vs истинные
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.scatter(y_val, y_pred, alpha=0.3, s=10)
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], 'r--', lw=2)
plt.xlabel('Истинные значения')
plt.ylabel('Предсказанные')
plt.title('Предсказанные vs истинные')

# 2. График остатков
plt.subplot(1,2,2)
plt.scatter(y_pred, residuals, alpha=0.3, s=10)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('Предсказанные значения')
plt.ylabel('Остатки')
plt.title('Остатки vs предсказанные')
plt.tight_layout()
plt.show()

# 3. Гистограмма остатков
plt.figure(figsize=(8,4))
plt.hist(residuals, bins=30, edgecolor='black')
plt.xlabel('Остатки')
plt.ylabel('Частота')
plt.title('Распределение остатков')
plt.show()

# 4. Q-Q plot (проверка нормальности остатков)
import scipy.stats as stats
plt.figure(figsize=(6,6))
stats.probplot(residuals, dist="norm", plot=plt)
plt.title('Q-Q plot остатков')
plt.show()

# 5. Важность признаков (коэффициенты линейной регрессии)
coeff = pd.Series(model.coef_, index=features)
coeff_sorted = coeff.abs().sort_values(ascending=False)
plt.figure(figsize=(10,6))
coeff_sorted.head(15).plot(kind='bar')
plt.title('Абсолютные значения коэффициентов (важность признаков)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 6. Зависимость остатков от наиболее важного признака (foot_traffic)
plt.figure(figsize=(8,5))
plt.scatter(X_val['foot_traffic'], residuals, alpha=0.3, s=10)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel('foot_traffic')
plt.ylabel('Остатки')
plt.title('Остатки vs foot_traffic')
plt.show()