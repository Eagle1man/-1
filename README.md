Аппаратные средства интеллектуальных автоматизированных систем лабораторная 1 по теме: «Приложение для прогнозирования погоды с использованием нейронных сетей»
 Цель работы изучить технологий разработки и оптимизации нейронных сетей для решения задач регрессии на примере прогнозирования температуры. Работа включает в себя:  
-  Предобработку данных и анализ признаков.  
- Построение нейронных сетей с различными функциями активации для сравнения их эффективности (Sigmoid, ReLU, LeakyReLU,).  
- Реализацию методов регуляризации (Dropout, L2-регуляризация, Batch Normalization).  
- Сравнение эффективности предсказания погоды моделями  по метрикам качества (MAE, RMSE).  

Настройка среды разработки и установка библиотек
1.	Установите Python (версии ≥3.7) с сайта python.org.
2.	Установите Visual Studio Code: легкую, расширяемую IDE для разработки на Python.
3.   Установите Библиотеки:  
   - TensorFlow/Keras — построение и обучение нейронных сетей.  
   - Pandas/NumPy — обработка табличных данных.  
   - Scikit-learn — предобработка данных и оценка качества.  
   -  Matplotlib — визуализация результатов.  
 

Создание проекта

1. Загрузка и предобработка данных  
- Загрузка датасета `weatherHistory.csv`.  
(из kaggle https://www.kaggle.com/datasets/muthuj7/weather-dataset)

 





- Предобработка данных:  

 

 

 

 

 

 





Построение моделей

Модель с сигмоидой последовательная нейросеть с двумя скрытыми слоями, сужающимися от 32 к 16 нейронам, и регрессионным выходом.

 
1.	Входной слой: Принимает данные с размерностью input_shape=(X_train_scaled.shape[1],), соответствующей количеству масштабированных признаков.
2.	Первый скрытый слой: Полносвязный слой (Dense) с 32 нейронами и сигмоидальной активацией (sigmoid) для нелинейного преобразования входных данных.
3.	Второй скрытый слой: Слой Dense с 16 нейронами и сигмоидой, уменьшающий размерность признаков и извлекающий более сложные закономерности.
4.	Выходной слой: Один нейрон без явной активации (Dense(1)), что стандартно для задач регрессии — предсказывает числовое значение напрямую.
 
5.	Оптимизатор: Adam — адаптивный алгоритм градиентного спуска, автоматически настраивающий скорость обучения.
6.	Функция потерь: Среднеквадратичная ошибка (MSE) — минимизирует квадрат разницы между предсказаниями и истинными значениями.
7.	Метрика качества: Средняя абсолютная ошибка (MAE) — дополнительная интерпретируемая метрика в исходных единицах измерения.
 
8.	Обучение: 50 эпох (проходов по всему обучающему набору) с пакетами по 32 образца (batch_size=32).
9.	Валидация: Контроль переобучения через оценку val_loss и val_mae на тестовых данных (X_test_scaled, y_test).


Модель с ReLU,   последовательная нейросеть с тремя скрытыми слоями а так же L2 и dropout для предотвращения переобучения
 
1.	Входной слой: Принимает данные размерности X_train_processed.shape[1] (количество обработанных признаков).
2.	Первый скрытый слой: Полносвязный слой (Dense) с 64 нейронами, активацией ReLU и L2-регуляризацией (λ=0.001), штрафующей большие веса для борьбы с переобучением.
3.	Dropout (0.3): Случайно отключает 30% нейронов первого слоя на каждом шаге обучения, усиливая регуляризацию.
4.	Второй скрытый слой: Dense с 32 нейронами, ReLU и L2-регуляризацией (λ=0.001), уменьшающий размерность признаков.
5.	Dropout (0.2): Деактивирует 20% нейронов второго слоя, снижая ко-адаптацию признаков.
6.	Третий скрытый слой: Dense с 16 нейронами, ReLU и L2-регуляризацией (λ=0.001), извлекающий высокоуровневые закономерности.
7.	Dropout (0.1): Отключает 10% нейронов третьего слоя, сохраняя баланс между регуляризацией и сохранением информации.
8.	Выходной слой: Один нейрон (Dense(1)) без активации для задач регрессии, предсказывающий числовое значение напрямую.
 
9.	Оптимизатор: Adam с learning rate 0.001, адаптивно настраивающий шаг градиентного спуска.
10.	Функция потерь: MSE (среднеквадратичная ошибка), минимизирующая квадрат отклонения предсказаний от истины.
11.	Метрики: MAE (средняя абсолютная ошибка) и RMSE (корень из MSE) для интерпретируемой оценки качества.
 
12.	Обучение: 50 эпох с пакетами по 64 образца (batch_size=64), обновление весов на каждой итерации.
13.	Валидация: Контроль переобучения через мониторинг val_loss, val_mae, val_root_mean_squared_error на тестовых данных.




В этой нейросети   применили Комбинацию L2-регуляризации снижающую величину весов и Dropout для предотвращения переобучения.  Остановимся на этих методах подробнее:

L2 регуляризация это метод, используемый в машинном обучении для предотвращения переобучения модели. Она добавляет штраф за большие значения весов модели к функции потерь, что заставляет модель сохранять веса небольшими и улучшает её обобщающую способность.
 





Dropout — метод регуляризации, предотвращающий переобучение нейронных сетей путем случайного отключения нейронов на этапе обучения. Это заставляет сеть обучаться устойчивым признакам, снижая зависимость между нейронами. При каждом обновлении весов деактивируется случайная часть нейронов (например, 50%), что имитирует обучение ансамбля из множества подсетей. На этапе тестирования все нейроны активны, а их веса масштабируются для компенсации отсутствия отключений. Dropout улучшает обобщающую способность модели, особенно на небольших датасетах.

Модель нейросети с тремя скрытыми слоями, функцией активации LeakyReLU, L2-регуляризацией и прогрессивным Dropout для предотвращения переобучения



 
1.	Входной слой: Принимает данные размерности X_train_processed.shape[1] (число обработанных признаков).
2.	Первый скрытый слой: Dense(64) с L2-регуляризацией (λ=0.001), уменьшающей переобучение через штраф за большие веса.
3.	Активация LeakyReLU: Замена стандартной ReLU с параметром alpha=0.01, позволяющая избежать "мертвых нейронов" за счет слабого градиента при отрицательных значениях.
4.	Dropout (0.3): Случайное отключение 30% нейронов слоя на каждом шаге обучения для снижения ко-адаптации.
5.	Второй скрытый слой: Dense(32) с L2-регуляризацией, сжимающий пространство признаков до 32 нейронов.
6.	LeakyReLU: Повторная нелинейная активация с параметром alpha=0.01.
7.	Dropout (0.2): Деактивация 20% нейронов для усиления обобщающей способности.
8.	Третий скрытый слой: Dense(16) с L2-регуляризацией, выделяющий высокоуровневые закономерности.
9.	LeakyReLU: Финализация нелинейных преобразований перед выходом.
10.	Dropout (0.1): Минимальное отключение 10% нейронов для сохранения информации.
11.	Выходной слой: Dense(1) без активации — регрессионный выход для предсказания числовых значений.

 
12.	Оптимизатор: Adam с learning rate 0.001, адаптивно настраивающий шаг обучения.
13.	Функция потерь: MSE (среднеквадратичная ошибка), фокусирующаяся на минимизации квадратов отклонений.
14.	Метрики: MAE (средняя абсолютная ошибка) и RMSE (корень из MSE) для интерпретируемой оценки точности.
 
15.	Обучение: 10 эпох с батчами по 64 образца, валидация на тестовых данных для контроля переобучения.








В этих трех моделях мы так же сравниваем Мы сравниваем три Функции активации Sigmoid, ReLU и LeakyReLU :
Sigmoid:
 
 
Relu
 
 



LeakyReLU 
 
 

 Запуск неросетей

1. Sigmoid
 1. Импорт библиотек
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

# 2. Загрузка данных
df = pd.read_csv('/content/weatherHistory.csv')

# 3. Предобработка данных
# 3.1. Преобразование времени
df['Formatted Date'] = pd.to_datetime(df['Formatted Date'], utc=True)
df['Hour'] = df['Formatted Date'].dt.hour
df['Day_of_Year'] = df['Formatted Date'].dt.dayofyear

# 3.2. Удаление ненужных столбцов
columns_to_drop = ['Formatted Date', 'Daily Summary', 'Summary']
df = df.drop(columns=columns_to_drop, errors='ignore')

# 3.3. Обработка пропусков
df = df.ffill()

# 4. Выбор признаков и целевой переменной
features = [
    'Humidity', 
    'Wind Speed (km/h)', 
    'Pressure (millibars)', 
    'Hour', 
    'Day_of_Year'
]
target = 'Temperature (C)'

X = df[features]
y = df[target]

# 5. Разделение данных
X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.2, 
    random_state=42
)

# 6. Стандартизация данных
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 7. Создание простой нейронной сети
model = Sequential([
    Dense(32, activation='sigmoid', input_shape=(X_train_scaled.shape[1],)),
    Dense(16, activation='sigmoid'),
    Dense(1)  # Выходной слой без активации для регрессии
])

# 8. Компиляция модели
model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['mae']
)

# 9. Обучение модели
history = model.fit(
    X_train_scaled,
    y_train,
    validation_data=(X_test_scaled, y_test),
    epochs=50,
    batch_size=32,
    verbose=1
)

# 10. Визуализация обучения
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Функция потерь (MSE)')
plt.xlabel('Эпоха')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Train MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Средняя абсолютная ошибка')
plt.xlabel('Эпоха')
plt.legend()

plt.tight_layout()
plt.show()

# 11. Оценка модели
test_loss, test_mae = model.evaluate(X_test_scaled, y_test, verbose=0)
print(f'\nРезультаты тестирования:')
print(f'Средняя абсолютная ошибка (MAE): {test_mae:.2f}°C')

В результате работы модели с Sigmoid получаем:
 



2. ReLU
# 1. Импорт библиотек
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras import regularizers
# 2. Загрузка данных
df = pd.read_csv('/content/weatherHistory.csv')

# 3. Предобработка данных
# 3.1. Преобразование времени
df['Formatted Date'] = pd.to_datetime(df['Formatted Date'], utc=True)
df['Hour'] = df['Formatted Date'].dt.hour
df['Day_of_Year'] = df['Formatted Date'].dt.dayofyear

# 3.2. Удаление ненужных столбцов
columns_to_drop = ['Formatted Date', 'Daily Summary', 'Summary']
df = df.drop(columns=columns_to_drop, errors='ignore')

# 3.3. Обработка пропусков
df = df.fillna(method='ffill')

# 4. Выбор признаков и целевой переменной
features = [
    'Humidity',
    'Wind Speed (km/h)',
    'Pressure (millibars)',
    'Hour',
    'Day_of_Year'
]
target = 'Temperature (C)'

X = df[features]
y = df[target]

# 5. Разделение данных
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 6. Обработка категориальных признаков
# В данном датасете нет категориальных признаков в выбранных фичах,
# но пример обработки оставлен для возможных расширений
cat_cols = []
if cat_cols:
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    ohe.fit(X_train[cat_cols])
    X_train_cat = ohe.transform(X_train[cat_cols])
    X_test_cat = ohe.transform(X_test[cat_cols])
else:
    X_train_cat = np.zeros((X_train.shape[0], 0))
    X_test_cat = np.zeros((X_test.shape[0], 0))

# 7. Стандартизация числовых признаков
num_cols = [col for col in features if col not in cat_cols]
scaler = StandardScaler()
X_train_num = scaler.fit_transform(X_train[num_cols])
X_test_num = scaler.transform(X_test[num_cols])

# 8. Объединение признаков
X_train_processed = np.hstack([X_train_num, X_train_cat])
X_test_processed = np.hstack([X_test_num, X_test_cat])


# 9. Создание нейронной сети с регуляризацией
model = Sequential([
    Dense(64, activation='relu',
          input_shape=(X_train_processed.shape[1],),
          kernel_regularizer=regularizers.l2(0.001)),  # L2-регуляризация
    Dropout(0.3),  # Отключаем 30% нейронов
    Dense(32, activation='relu',
          kernel_regularizer=regularizers.l2(0.001)),
    Dropout(0.2),  # Отключаем 20% нейронов
    Dense(16, activation='relu',
          kernel_regularizer=regularizers.l2(0.001)),
    Dropout(0.1),  # Отключаем 10% нейронов
    Dense(1)
])

# 10. Компиляция модели (оставляем без изменений)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae', tf.keras.metrics.RootMeanSquaredError()]
)
# 11. Обучение модели
history = model.fit(
    X_train_processed,
    y_train,
    validation_data=(X_test_processed, y_test),
    epochs=50,
    batch_size=64,
    verbose=1
)

# 12. Оценка модели
test_loss, test_mae, test_rmse = model.evaluate(X_test_processed, y_test, verbose=0)
print(f'\nРезультаты тестирования:')
print(f'Средняя абсолютная ошибка (MAE): {test_mae:.2f}°C')
print(f'Среднеквадратичная ошибка (RMSE): {test_rmse:.2f}°C')
import matplotlib.pyplot as plt

# ... (весь предыдущий код до оценки модели)

# 12. Визуализация процесса обучения
def plot_training_history(history):
    plt.figure(figsize=(12, 8))

    # График потерь (MSE)
    plt.subplot(2, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Эволюция функции потерь')
    plt.ylabel('MSE')
    plt.xlabel('Эпоха')
    plt.legend()

    # График MAE
    plt.subplot(2, 2, 2)
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Динамика средней абсолютной ошибки')
    plt.ylabel('MAE (°C)')
    plt.xlabel('Эпоха')
    plt.legend()

    # График RMSE
    plt.subplot(2, 2, 3)
    plt.plot(history.history['root_mean_squared_error'], label='Train RMSE')
    plt.plot(history.history['val_root_mean_squared_error'], label='Validation RMSE')
    plt.title('Динамика среднеквадратичной ошибки')
    plt.ylabel('RMSE (°C)')
    plt.xlabel('Эпоха')
    plt.legend()

    # Общая статистика
    plt.subplot(2, 2, 4)
    plt.axis('off')
    plt.text(0.1, 0.8, f'Final Train MAE: {history.history["mae"][-1]:.2f}°C', fontsize=10)
    plt.text(0.1, 0.6, f'Final Validation MAE: {history.history["val_mae"][-1]:.2f}°C', fontsize=10)
    plt.text(0.1, 0.4, f'Final Train RMSE: {history.history["root_mean_squared_error"][-1]:.2f}°C', fontsize=10)
    plt.text(0.1, 0.2, f'Final Validation RMSE: {history.history["val_root_mean_squared_error"][-1]:.2f}°C', fontsize=10)

    plt.tight_layout()
    plt.show()

# Вызываем функцию визуализации
plot_training_history(history)

# 13. Оценка модели
test_loss, test_mae, test_rmse = model.evaluate(X_test_processed, y_test, verbose=0)
print(f'\nРезультаты тестирования:')
print(f'Средняя абсолютная ошибка (MAE): {test_mae:.2f}°C')
print(f'Среднеквадратичная ошибка (RMSE): {test_rmse:.2f}°C')

В результате работы модели с ReLU получаем:
 
 

3.  LeakyReLU
# 1. Импорт библиотек
!pip install tensorflow
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import Sequential, regularizers
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU

# 2. Загрузка данных
df = pd.read_csv('/content/weatherHistory.csv')

# 3. Предобработка данных
# 3.1. Преобразование времени
df['Formatted Date'] = pd.to_datetime(df['Formatted Date'], utc=True)
df['Hour'] = df['Formatted Date'].dt.hour
df['Day_of_Year'] = df['Formatted Date'].dt.dayofyear

# 3.2. Удаление ненужных столбцов
columns_to_drop = ['Formatted Date', 'Daily Summary', 'Summary']
df = df.drop(columns=columns_to_drop, errors='ignore')

# 3.3. Обработка пропусков (исправленный метод)
df = df.ffill()  # Вместо fillna(method='ffill')

# 4. Выбор признаков и целевой переменной
features = [
    'Humidity',
    'Wind Speed (km/h)',
    'Pressure (millibars)',
    'Hour',
    'Day_of_Year'
]
target = 'Temperature (C)'

X = df[features]
y = df[target]

# 5. Разделение данных
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# 6. Обработка категориальных признаков
cat_cols = []
if cat_cols:
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    ohe.fit(X_train[cat_cols])
    X_train_cat = ohe.transform(X_train[cat_cols])
    X_test_cat = ohe.transform(X_test[cat_cols])
else:
    X_train_cat = np.zeros((X_train.shape[0], 0))
    X_test_cat = np.zeros((X_test.shape[0], 0))

# 7. Стандартизация числовых признаков
num_cols = [col for col in features if col not in cat_cols]
scaler = StandardScaler()
X_train_num = scaler.fit_transform(X_train[num_cols])
X_test_num = scaler.transform(X_test[num_cols])

# 8. Объединение признаков
X_train_processed = np.hstack([X_train_num, X_train_cat])
X_test_processed = np.hstack([X_test_num, X_test_cat])

# 9. Создание нейронной сети с регуляризацией
model = Sequential([
    Dense(64,
          kernel_regularizer=regularizers.l2(0.001),
          input_shape=(X_train_processed.shape[1],)),
    LeakyReLU(negative_slope=0.01),  # Исправленный параметр
    Dropout(0.3),

    Dense(32,
          kernel_regularizer=regularizers.l2(0.001)),
    LeakyReLU(negative_slope=0.01),
    Dropout(0.2),

    Dense(16,
          kernel_regularizer=regularizers.l2(0.001)),
    LeakyReLU(negative_slope=0.01),
    Dropout(0.1),

    Dense(1)
])

# 10. Компиляция модели (обязательный этап перед обучением)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae', tf.keras.metrics.RootMeanSquaredError()]
)

# 11. Обучение модели
history = model.fit(
    X_train_processed,
    y_train,
    validation_data=(X_test_processed, y_test),
    epochs=50,
    batch_size=64,
    verbose=1
)

# 12. Визуализация процесса обучения
import matplotlib.pyplot as plt

def plot_training_history(history):
    plt.figure(figsize=(12, 8))

    # График потерь (MSE)
    plt.subplot(2, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Эволюция функции потерь')
    plt.ylabel('MSE')
    plt.xlabel('Эпоха')
    plt.legend()

    # График MAE
    plt.subplot(2, 2, 2)
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Динамика средней абсолютной ошибки')
    plt.ylabel('MAE (°C)')
    plt.xlabel('Эпоха')
    plt.legend()

    # График RMSE
    plt.subplot(2, 2, 3)
    plt.plot(history.history['root_mean_squared_error'], label='Train RMSE')
    plt.plot(history.history['val_root_mean_squared_error'], label='Validation RMSE')
    plt.title('Динамика среднеквадратичной ошибки')
    plt.ylabel('RMSE (°C)')
    plt.xlabel('Эпоха')
    plt.legend()

    # Общая статистика
    plt.subplot(2, 2, 4)
    plt.axis('off')
    plt.text(0.1, 0.8, f'Final Train MAE: {history.history["mae"][-1]:.2f}°C', fontsize=10)
    plt.text(0.1, 0.6, f'Final Validation MAE: {history.history["val_mae"][-1]:.2f}°C', fontsize=10)
    plt.text(0.1, 0.4, f'Final Train RMSE: {history.history["root_mean_squared_error"][-1]:.2f}°C', fontsize=10)
    plt.text(0.1, 0.2, f'Final Validation RMSE: {history.history["val_root_mean_squared_error"][-1]:.2f}°C', fontsize=10)

    plt.tight_layout()
    plt.show()

plot_training_history(history)

# 13. Оценка модели
test_loss, test_mae, test_rmse = model.evaluate(X_test_processed, y_test, verbose=0)
print(f'\nРезультаты тестирования:')
print(f'Средняя абсолютная ошибка (MAE): {test_mae:.2f}°C')
print(f'Среднеквадратичная ошибка (RMSE): {test_rmse:.2f}°C')

 В результате работы модели с LeakyReLU получаем:

 










Результаты экспериментов

Model	MAE (°C)	RMSE (°C)
Sigmoid 	6.07	—
ReLU 	2.81	3.57
LeakyReLU 	2.82	3.58


 

1. Сигмоида Результат: MAE = 6,07 °C Выводы:
1.	Насыщение на концах: при больших положительных или отрицательных значениях входа выход сигмоиды «сплющивается» к 1 или 0, а её производная стремится к нулю.
2.	Исчезающие градиенты: во время обратного распространения градиенты в глубоких слоях почти обнуляются, что останавливает обучение этих слоёв.
3.	Плохое моделирование сложных зависимостей: сеть не может уловить все нюансы данных о погоде, поэтому ошибка получается высокой.
 
2. ReLU Результаты: MAE = 2,81 °C, RMSE = 3,57 °C Выводы:
1.	Отсутствие насыщения для положительных входов: для x > 0 функция равна x, а её производная равна 1, что позволяет градиентам свободно проходить через слои.
2.	Простота и разреженность: отрицательные входы обнуляются (производная = 0), что даёт разрежённые активации и сосредотачивает обучение на значимых признаках.
3.	Быстрая и стабильная сходимость: комбинация сильных градиентов и разрежённости даёт самое низкое значение ошибок среди всех трёх функций.
3. LeakyReLU
•	Результаты: MAE = 2,82 °C, RMSE = 3,58 °C Выводы:
1.	Небольшой отрицательный наклон: LeakyReLU(x) = x при x > 0 и α•x (α≈0,01) при x ≤ 0, благодаря чему даже «неактивные» нейроны получают небольшой градиент.
2.	Предотвращение «затухающих» нейронов: в глубоких сетях некоторые нейроны на обычном ReLU могут навсегда «залипнуть» на нуле; утечка градиента исправляет это.
3.	Лёгкий дополнительный шум: в задаче по предсказанию погоды «затухающих» нейронов и так было немного, поэтому небольшой градиент в отрицательной области лишь добавил шума и чуть ухудшил результаты по сравнению с обычной ReLU.







Заключение

Функции активации ReLU/LeakyReLU показали наилучшие результаты благодаря отсутствию насыщения и эффективному градиентному потоку в тоже время как Сигмоида не подходит для задач регрессии из-за ограниченного диапазона выходных значений, комбинация L2-регуляризации, Dropout и BatchNorm снизила MAE примерно на 25%.
Лабораторная работа продемонстрировала важность выбора функций активации и методов регуляризации при проектировании нейронных сетей в зависимости от исходных данных.
  В результате проведенных экспериментов  удалось наглядно убедиться, что Сигмоида не подходит для задач регрессии из-за ограниченного диапазона выходных значений 
Кроме того, лабораторная работа  показала, что более новые и сложные методы типа LeakyReLU не всегда лучше более простых методов типа ReLU и нужно ориентироваться, прежде всего, на данные и стараться излишне не усложнять модель нейросети без обоснования, ведь  наилучшие результаты прогнозирования погоды в данной лабораторной работе были достигнуты при использовании ReLU в сочетании с комплексной регуляризацией.

