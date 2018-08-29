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

# K-Means 評估視覺化 (畫子圖)
def evaluate_axes_plot(title, axes, n_clusters_list, avgs):
	axes.plot(n_clusters_list, avgs)
	axes.set_title(title, fontsize=20)
	axes.set_xticks(n_clusters_list)
	axes.set_xticklabels(n_clusters_list)
	axes.set_xlabel('clusters', fontsize=12)

# 儲存 K-Means 評估視覺化圖
def save_evaluate_visualization(min_clusters, max_clusters):
	current_time = user_load_data.get_current_time()
	file_name = 'K-Means_evaluate_cluster_{}-{}_visualization.svg'.format(min_clusters, max_clusters)
	img_path = 'img/{}_{}'.format(current_time, file_name)
	plt.savefig(img_path, dpi=150)

# K-Means 評估視覺化
def evaluate_visualization(dataSet, min_clusters, max_clusters):
	if (max_clusters < 5):
		print('max_clusters 要大於 5')
		return

	silhouette_avgs = []
	calinski_harabaz_avgs = []
	peroid_column = user_load_data.create_peroid_column()
	tmp_dataSet = dataSet[peroid_column]

	n_clusters_list = range(min_clusters, max_clusters + 1)
	n_clusters_list_len = len(n_clusters_list)

	for n_clusters in n_clusters_list:
		kmeans_fit = KMeans(n_clusters=n_clusters)
		kmeans_fit.fit(tmp_dataSet)
		# 評估標準：Silhouette Coefficient
		silhouette_avg = metrics.silhouette_score(tmp_dataSet, kmeans_fit.labels_)
		silhouette_avgs.append(silhouette_avg)
		# 評估標準：Calinski-Harabasz Index
		calinski_harabaz_avg = metrics.calinski_harabaz_score(tmp_dataSet, kmeans_fit.labels_)
		calinski_harabaz_avgs.append(calinski_harabaz_avg)

	ncols, nrows = 2, 1
	figsize_x = (3 * n_clusters_list_len // 6 * ncols) + 4
	figsize_y = 2 * n_clusters_list_len // 8 * nrows
	fig, (ax1, ax2) = plt.subplots(nrows, ncols, figsize=(figsize_x, figsize_y))

	# 將評估標準視覺化
	evaluate_axes_plot('Silhouette Coefficient', ax1,
							  n_clusters_list, silhouette_avgs)
	evaluate_axes_plot('Calinski-Harabasz Index', ax2,
							  n_clusters_list, calinski_harabaz_avgs)
	# 儲存 K-Means 評估視覺化圖
	save_evaluate_visualization(min_clusters, max_clusters)

#	 print('Silhouette Coefficient:', silhouette_avgs)
#	 print('Calinski-Harabasz Index:', calinski_harabaz_avgs)