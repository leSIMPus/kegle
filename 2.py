import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler

#1. Загрузка датасета
df = pd.read_csv("./data/exams.csv")

print("Первые строки таблицы:")
print(df.head())
print("\nИнформация о данных:")
print(df.info())

#2. Визуализация распределения целевого признака
plt.figure(figsize=(8, 4))
sns.histplot(df['math score'], bins=15, kde=True)
plt.title("Распределение результатов по математике")
plt.xlabel("Баллы")
plt.ylabel("Количество учеников")
plt.show()

#3. Проверка и обработка пропусков
print("\nПроверка на пропущенные значения:")
print(df.isnull().sum())

if df.isnull().sum().sum() == 0:
    print("\nОшибка: в данных нет пропусков. Создаём искусственные пропуски для демонстрации.")
    df.loc[df.sample(frac=0.01).index, 'reading score'] = np.nan

print("\nПосле генерации пропусков:")
print(df.isnull().sum())

df['reading score'].fillna(df['reading score'].mean(), inplace=True)

#4. Преобразование данных
cat_cols = df.select_dtypes(include=['object']).columns
print(f"\nКатегориальные признаки: {cat_cols.tolist()}")

encoder = LabelEncoder()
for col in cat_cols:
    df[col] = encoder.fit_transform(df[col])

#5. Визуализация корреляций
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Корреляция признаков")
plt.show()

#6. Стандартизация числовых признаков
num_cols = df.select_dtypes(include=['int64', 'float64']).columns

scaler = StandardScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

print("\nПосле стандартизации:")
print(df.head())

print("\nВсе данные числовые и без пропусков:")
print(df.info())
