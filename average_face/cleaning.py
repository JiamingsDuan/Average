# ---* 数据清洗 *--
import pandas as pd

dataset_path = 'data/data_664.csv'
data = pd.read_csv(dataset_path)
dataset = data.dropna(axis=0, how='any').reset_index(drop=True)

dataset.to_csv('data/data_657.csv', sep=',', encoding='utf-8', index=False)
