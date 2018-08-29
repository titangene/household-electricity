import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from . import user_load_data

# 先做 channelId 0
def only_use_channelId_0_dataSet(dataSet):
	dataSet = dataSet[dataSet['channelId'] == 0]
	return dataSet

# 刪除異常值，因為發現 sensor 本身有問題
def delete_outliers_dataSet(dataSet):
	dataSet = user_load_data.delete_outliers(dataSet)
	# 改變 'reporttime' 欄位 type (string to datetime)
	dataSet = user_load_data.transform_time(dataSet, column='reportTime', format='%Y-%m-%d %H:%M:%S')
	return dataSet

# 以 userId 分類，彙整每個使用者用電資料為每 15 分鐘一筆，w 四捨五入至小數 2 位
def group_dataSet(dataSet):
	dataSet = user_load_data.groupbyData(dataSet, 'userId')
	return dataSet

# 彙整與轉置多個使用者的用電資料 (96 期)
def consolidation_dataSet(dataSet):
	# 建立彙整資料欄位
	peroid_column = user_load_data.create_consolidation_column()
	consolidation_dataSet_list = user_load_data.consolidation_all_dataSet(dataSet)
	dataSet = pd.DataFrame(consolidation_dataSet_list, columns=peroid_column)
	return dataSet

# 缺值處理
def process_na_dataSet(dataSet):
	# 建立彙整資料欄位
	peroid_column = user_load_data.create_consolidation_column()
	dataSet = user_load_data.process_na(dataSet, peroid_column, threshold=2)
	return dataSet

# 計算 最大需量、最大需量、總用電量
def calc_peroid_max_min_sum_w(dataSet):
	dataSet = user_load_data.peroid_max_min_sum_w(dataSet)
	return dataSet

def load_preprocess_dataSet(file_path):
    dateparse = lambda x: pd.datetime.strptime(x, '%Y-%m-%d')
    dataSet = user_load_data.load_dataset(file_path, dtype={ 'uuid': str, 'userId': str },
                                          date_parser=dateparse, parse_dates=['reportTime'])
    dataSet.set_index('uuid', inplace=True)
    return dataSet

def get_peroid_column_dataSet(dataSet):
    peroid_column = user_load_data.create_peroid_column()
    dataSet = dataSet[['userId'] + peroid_column][:1000]
    return dataSet