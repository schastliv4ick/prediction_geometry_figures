import os
import cv2
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import os

def extract_hu_features(img_path):
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((3,3), np.uint8)
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros(10)  # fallback
    largest = max(contours, key=cv2.contourArea)
    
    moments = cv2.moments(largest)
    hu = cv2.HuMoments(moments).flatten()
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
    
    area = cv2.contourArea(largest)
    perimeter = cv2.arcLength(largest, True)
    compactness = (perimeter**2) / area if area > 0 else 0
    epsilon = 0.02 * perimeter
    approx = cv2.approxPolyDP(largest, epsilon, True)
    n_vertices = len(approx)
    
    features = np.hstack([hu, [compactness, n_vertices, area/(64*64)]])
    return features

# --- Загрузка признаков для всего обучающего набора ---
df = pd.read_csv("train_labels.csv")
X_all = []
y_all = []
print("Извлечение признаков из train (все 12000)...")
for _, row in tqdm(df.iterrows(), total=len(df)):
    path = os.path.join("train", row['image_id'])
    feats = extract_hu_features(path)
    X_all.append(feats)
    y_all.append(row['class'])
X_all = np.array(X_all)
y_all = np.array(y_all)

# --- 5‑fold кросс‑валидация ---
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []
for fold, (train_idx, val_idx) in enumerate(skf.split(X_all, y_all)):
    X_tr, X_val = X_all[train_idx], X_all[val_idx]
    y_tr, y_val = y_all[train_idx], y_all[val_idx]
    
    rf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    pred_val = rf.predict(X_val)
    acc = accuracy_score(y_val, pred_val)
    cv_scores.append(acc)
    print(f"Fold {fold+1}: accuracy = {acc:.4f}")

print(f"\nСредняя точность по 5‑fold CV: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")

# --- Если средняя точность > 0.99, обучаем финальную модель на всех данных и предсказываем тест ---
if np.mean(cv_scores) > 0.99:
    print("\nМодель очень надёжна. Обучаем на всех train и предсказываем test...")
    final_rf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
    final_rf.fit(X_all, y_all)
    
    # Извлечение признаков для теста
    test_files = sorted([f for f in os.listdir("test") if f.endswith('.png')])
    X_test = []
    for fname in tqdm(test_files):
        path = os.path.join("test", fname)
        feats = extract_hu_features(path)
        X_test.append(feats)
    X_test = np.array(X_test)
    pred_test = final_rf.predict(X_test)
    
    # Сохранение submission
    sub = pd.DataFrame({'image_id': test_files, 'class': pred_test})
    sub.to_csv("submission.csv", index=False)
    print("✅ submission.csv сохранён. Загружайте его в систему проверки.")
else:
    print("Точность ниже ожидаемой. Нужно улучшать модель.")