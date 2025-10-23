import os
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi

# --- 1. Авторизация Kaggle API ---
api = KaggleApi()
api.authenticate()

# --- 2. Скачиваем датасет с Kaggle ---
dataset = 'whenamancodes/students-performance-in-exams'
api.dataset_download_files(dataset, path='./data', unzip=True)

# --- 3. Загружаем CSV ---
csv_files = [f for f in os.listdir('./data') if f.endswith('.csv')]
csv_path = os.path.join('./data', csv_files[0])

df = pd.read_csv(csv_path)

# --- 4. Просмотр данных ---
print(f"Размер датасета: {df.shape}")
print(f"Колонки: {df.columns.tolist()}")
print(df.head())

# --- 5. Выделение целевого признака ---
# Пусть целевой признак – результат по математике (math score)
target = df["math score"]

print("\nПример значений целевого признака (math score):")
print(target.head())
