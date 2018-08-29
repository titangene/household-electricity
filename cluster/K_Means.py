import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn import metrics

from preprocess import user_load_data

# 將 K-Means 每個集群中心的坐標資料存成 dataFrame
def get_cluster_centers_dataFrame(cluster_centers):
	peroid_column = user_load_data.create_peroid_column()
	# 將資料存成 DataFrame，並將數值做四捨五入
	centroids_df = pd.DataFrame(cluster_centers, columns=peroid_column).round(2)
	# 增加 Index
	centroids_df['index'] = list(range(0, len(centroids_df)))
	# 重新排列欄位順序，讓 Index 欄位變成最前面
	centroids_df = centroids_df.reindex(sorted(centroids_df.columns), axis=1)
	return centroids_df

# 將 K-Means 每個集群中心的坐標資料資料存成 CSV
def save_cluster_centers_csv(dataSet, n_clusters):
	file_name = 'K-Means_{}-cluster_centers.csv'.format(n_clusters)
	user_load_data.save_csv(dataSet, file_name)