import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from lightfm import utils
from lightfm.preprocessing import CSRFRecommender
from lightfm.evaluation import eval_precision

# Загрузка данных из CSV-файлов
train_data = pd.read_csv('train.csv')  # Данные о прослушиваниях в обучающем наборе
test_data = pd.read_csv('test.csv')  # Данные о прослушиваниях в тестовом наборе
songs_data = pd.read_csv('songs.csv')  # Информация о песнях
members_data = pd.read_csv('members.csv')  # Информация о пользователях
song_extra_info_data = pd.read_csv('song_extra_info.csv')  # Дополнительная информация о песнях

# Создание матрицы взаимодействия пользователь-песня
interactions = train_data[['msno', 'song_id']].groupby(['msno', 'song_id'])['target'].sum().unstack(fill_value=0)

# Преобразование ID пользователей и песен в числовые индексы
user_map = utils.mapping(interactions.index)  # Создание словаря для преобразования ID пользователей
song_map = utils.mapping(interactions.columns)  # Создание словаря для преобразования ID песен

# Преобразование матрицы взаимодействия в формат CSR (Compressed Sparse Row)
interactions_matrix = interactions.to_sparse(fill_value=0).asformat('csr')

# Разделение данных на обучающий и проверочный наборы
train_interactions, val_interactions = train_test_split(CSRFRecommender.prepare_interactions(interactions_matrix, user_map, song_map), test_size=0.2)

# Создание модели рекомендаций
model = CSRFRecommender(interactions_matrix=train_interactions, user_features=None, item_features=None)

# Обучение модели
model.fit(epochs=10, num_threads=4)  # Обучение модели на 10 эпох, используя 4 потока

# Оценка модели на проверочном наборе
precision_at_20 = eval_precision(model, val_interactions, k=20, num_threads=4)
print(f'NDCG@20 на проверочном наборе: {precision_at_20}')  # Вычисление NDCG@20 на проверочном наборе

# Генерация предсказаний на тестовом наборе
test_predictions = model.predict_all(user_ids=test_data['msno'], item_ids=test_data['song_id'])

# Подготовка файла с предсказаниями
submission_df = pd.DataFrame({'id': test_data['id'], 'target': test_predictions})
submission_df.to_csv('submission.csv', index=False)  # Сохранение предсказаний в файл submission.csv
