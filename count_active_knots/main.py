import cv2
import numpy as np
import pandas as pd
import os
import re

# ----------------------------------------------------------------------
# 1. Функция построения сетки 8x8 (наличие красных и синих в каждом блоке 16x16)
# ----------------------------------------------------------------------
def build_grid(img_path):
    """
    Принимает путь к изображению 128x128.
    Возвращает две бинарные матрицы 8x8: red_grid, blue_grid.
    red_grid[i,j] = True, если в блоке (i,j) есть хотя бы один красный пиксель.
    blue_grid[i,j] = True, если в блоке (i,j) есть хотя бы один синий пиксель.
    """
    img = cv2.imread(img_path)          # BGR
    if img is None:
        # На случай ошибки чтения возвращаем пустые сетки
        return np.zeros((8,8), bool), np.zeros((8,8), bool)
    
    h, w = 128, 128
    block_size = 16                     # 128/16 = 8
    red = np.zeros((8,8), bool)
    blue = np.zeros((8,8), bool)
    
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            gi = i // block_size        # индекс строки в сетке 0..7
            gj = j // block_size        # индекс столбца в сетке 0..7
            roi = img[i:i+block_size, j:j+block_size]   # блок 16x16
            
            # Красный: R (канал 2) > 250, G и B < 5
            has_red = np.any((roi[:,:,2] >= 250) & (roi[:,:,1] <= 5) & (roi[:,:,0] <= 5))
            # Синий: B (канал 0) > 250, G и R < 5
            has_blue = np.any((roi[:,:,0] >= 250) & (roi[:,:,1] <= 5) & (roi[:,:,2] <= 5))
            
            if has_red:
                red[gi, gj] = True
            if has_blue:
                blue[gi, gj] = True
                
    return red, blue

# ----------------------------------------------------------------------
# 2. Подсчёт активных красных точек по правилу 8-соседей
# ----------------------------------------------------------------------
def count_active(red, blue):
    """
    Принимает бинарные матрицы 8x8.
    Возвращает количество красных клеток, у которых есть хотя бы один синий сосед
    (включая диагонали, т.е. 8-связность).
    """
    active = 0
    for i in range(8):
        for j in range(8):
            if red[i, j]:
                # Проверяем всех 8 соседей
                found = False
                for di in (-1, 0, 1):
                    for dj in (-1, 0, 1):
                        if di == 0 and dj == 0:
                            continue
                        ni, nj = i + di, j + dj
                        if 0 <= ni < 8 and 0 <= nj < 8 and blue[ni, nj]:
                            found = True
                            break
                    if found:
                        break
                if found:
                    active += 1
    return active

# ----------------------------------------------------------------------
# 3. (Опционально) Оценка на обучающем наборе, если есть разметка
# ----------------------------------------------------------------------
if os.path.exists("train_labels.csv"):
    print("Найден файл train_labels.csv – выполняем проверку на обучающей выборке...")
    df_train = pd.read_csv("train_labels.csv")
    errors = []
    total = len(df_train)
    
    # Для ускорения можно обрабатывать не все 8000, но для точности пройдём по всем
    for idx, row in df_train.iterrows():
        img_path = os.path.join("train", row['image_id'])
        r, b = build_grid(img_path)
        pred = count_active(r, b)
        errors.append(abs(pred - row['count']))
        if (idx + 1) % 1000 == 0:
            print(f"  Обработано {idx+1} из {total} изображений")
    
    mae = np.mean(errors)
    print(f"\nСредняя абсолютная ошибка (MAE) на обучающем наборе: {mae:.6f}")
    if mae < 0.001:
        print("  -> MAE = 0, алгоритм работает идеально на обучающих данных.")
    else:
        print("  -> Есть расхождения, проверьте реализацию.")
else:
    print("Файл train_labels.csv не найден – пропускаем проверку на обучении.")

# ----------------------------------------------------------------------
# 4. Формирование предсказаний для тестового набора
# ----------------------------------------------------------------------
# Список всех файлов .png в директории test, имена строго 6 цифр + .png
# (исключаем файлы с суффиксами типа (1) – они не должны попадать в сабмит)
test_files = [f for f in os.listdir("test") if re.match(r'^\d{6}\.png$', f)]
test_files.sort()
print(f"\nНайдено {len(test_files)} тестовых изображений. Обработка...")

submission = []
for idx, fname in enumerate(test_files):
    img_path = os.path.join("test", fname)
    r, b = build_grid(img_path)
    pred = count_active(r, b)
    submission.append([fname, pred])
    if (idx + 1) % 500 == 0:
        print(f"  Обработано {idx+1} из {len(test_files)}")

# ----------------------------------------------------------------------
# 5. Сохранение результата
# ----------------------------------------------------------------------
sub_df = pd.DataFrame(submission, columns=['image_id', 'count'])
sub_df.to_csv("submission.csv", index=False)
print("\nГотов файл submission.csv. Загружайте его в проверяющую систему.")