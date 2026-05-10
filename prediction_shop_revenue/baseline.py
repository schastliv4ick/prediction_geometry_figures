import pandas as pd
from sklearn.linear_model import LinearRegression

# Загрузка
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print("=== ПРОСТЕЙШИЙ BASELINE ===")
print(f"Train shape: {train.shape}, Test shape: {test.shape}")

# Признаки: всё, кроме id и target
features = [c for c in train.columns if c not in ['id', 'target']]
X = train[features]
y = train['target']
X_test = test[features]

print("Используемые признаки:", features)

# Обучение на всех train
model = LinearRegression()
model.fit(X, y)

# Предсказание на test
pred = model.predict(X_test)

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
pred_val = model.predict(X_val)
mae = mean_absolute_error(y_val, pred_val)
print(f"MAE на валидации: {mae:.2f}")