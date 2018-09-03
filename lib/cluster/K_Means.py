import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans, MiniBatchKMeans

from ..preprocess import user_load_data

# 將 K-Means 每個集群中心的坐標資料存成 dataFrame
def get_cluster_centers_dataFrame(cluster_centers):
	peroid_column = user_load_data.create_peroid_column()
	# 將資料存成 DataFrame，並將數值做四捨五入
	centroids_df = pd.DataFrame(cluster_centers, columns=peroid_column).round(2)
	# 增加 Index
	centroids_df['index'] = list(range(0, len(centroids_df)))
	# 重新排列欄位順序，讓 Index 欄位變成最前面
	peroid_and_index_column = ['index'] + peroid_column
	centroids_df = centroids_df.reindex(peroid_and_index_column, axis=1)
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

'''
將 K-Means 分群中各屬於該群資料和中心點畫成子圖，
一次 K-Means 分群會畫成一列 (有 n 群就會畫成 n 欄)

values:
- cluster_label: 該群標籤
- cluster_center: 該群中心點
- ax: 畫子圖用 (type: AxesSubplot)
grouped: 以 cluster (群集) 欄位 groupby 的某一群用電資料
n_clusters: K-Means 分成幾群
'''
def cluster_axes_plot(values, grouped, n_clusters, peroid_column):
	cluster_label, cluster_center, ax = values

	ax.plot(grouped.get_group(cluster_label).loc[:, peroid_column].T, alpha=0.13, color='gray')
	# red solid line and point marker
	ax.plot(cluster_center, 'r.-', alpha=0.5)

	ax.set_xlim(-1, 97)
	tick_locations = [0] + list(range(3, 97, 4))
	tick_labels =  [1] + list(range(4, 97, 4))
	ax.set_xticks(tick_locations)
	ax.set_xticklabels(tick_labels)

	ax.set_title('{} clusters: label_{}'.format(n_clusters, cluster_label), fontsize=32)
	ax.set_xlabel('period', fontsize=20)
	ax.set_ylabel('w', fontsize=20)

	print('n_clusters:{}, label:{}'.format(n_clusters, cluster_label))

'''
K-Means 分成 n 群，並畫出每群的資料和中心點

values:
- n_clusters: K-Means 分成幾群
- ax_row: 畫子圖的某一列 (type: ndarray, 內有多個 AxesSubplot)
dataSet: 彙整的用電資料
peroid_column: 以日為單位之欄位 (96 期)，['period_1', ... , 'period_96']
'''
def nxm_clusters_plot(values, dataSet, peroid_column):
	n_clusters, ax_row = values
	kmeans_fit = KMeans(n_clusters=n_clusters)
	kmeans_fit.fit(dataSet.loc[:, peroid_column])

	dataSet['cluster'] = kmeans_fit.labels_
	grouped = dataSet.groupby('cluster')
	# 分群後，刪除 cluster 欄位
	dataSet.drop(['cluster'], axis=1, inplace=True)
	targets = zip(grouped.groups.keys(), kmeans_fit.cluster_centers_, ax_row)
	# 將 K-Means 分群中各屬於該群資料和中心點畫成子圖，
	# 一次 K-Means 分群會畫成一列 (有 n 群就會畫成 n 欄)
	axes_plot_func = lambda values: cluster_axes_plot(values, grouped, n_clusters, peroid_column)
	list(map(axes_plot_func, targets))

# 儲存 K-Means n ~ m 分群的視覺化圖
def save_nxm_clusters_visualization(min_clusters, max_clusters):
	current_time = user_load_data.get_current_time()
	file_name = 'K-Means_cluster_{}-{}_visualization.svg'.format(min_clusters, max_clusters)
	img_path = 'img/{}_{}'.format(current_time, file_name)
	plt.savefig(img_path, dpi=150)

# K-Means n ~ m 分群的視覺化圖
def nxm_clusters_visualization(dataSet, min_clusters, max_clusters):
	peroid_column = user_load_data.create_peroid_column()
	# 建立 (min_clusters ~ max_clusters) 的群集數 list
	n_clusters_list = range(min_clusters, max_clusters + 1)
	ncols, nrows = max_clusters, len(n_clusters_list)

	fig, axes = plt.subplots(nrows, ncols, figsize=(20 * ncols, 6 * nrows),
							 gridspec_kw=dict(hspace=0.5, wspace=0.12))
	# colors = generate_colors(max_clusters)

	targets = zip(n_clusters_list, axes)
	# K-Means 分成 n 群，並畫出每群的資料和中心點
	plot_func = lambda values: nxm_clusters_plot(values, dataSet, peroid_column)
	list(map(plot_func, targets))
	# 儲存 K-Means n ~ m 分群的視覺化圖
	save_nxm_clusters_visualization(min_clusters, max_clusters)
	plt.show()

# 儲存 K-Means n 分群的視覺化矩陣圖
def save_n_clusters_visualization_matrix(n_clusters):
	current_time = user_load_data.get_current_time()
	file_name = 'K-Means_cluster_{}_visualization_matrix.svg'.format(n_clusters)
	img_path = 'img/{}_{}'.format(current_time, file_name)
	plt.savefig(img_path, dpi=150)

# K-Means n 分群的視覺化矩陣圖
def n_clusters_visualization_matrix(dataSet, n_clusters, ncols):
	peroid_column = user_load_data.create_peroid_column()
	kmeans_fit = KMeans(n_clusters=n_clusters).fit(dataSet[peroid_column])

	dataSet['cluster'] = kmeans_fit.labels_
	grouped = dataSet.groupby('cluster')
	# 分群後，刪除 cluster 欄位
	dataSet.drop(['cluster'], axis=1, inplace=True)

	if (n_clusters % ncols == 0):
		nrows = n_clusters // ncols
	else:
		nrows = (n_clusters // ncols) + 1
	fig, axes = plt.subplots(nrows, ncols, figsize=(20 * ncols, 6 * nrows),
							 gridspec_kw=dict(hspace=0.5, wspace=0.12))

	targets = zip(grouped.groups.keys(), kmeans_fit.cluster_centers_, axes.flatten())
	for idx, (cluster_label, cluster_center, ax) in enumerate(targets):
		ax.plot(grouped.get_group(cluster_label).loc[:, peroid_column].T, alpha=0.13, color='gray')
		# red solid line and point marker
		ax.plot(cluster_center, 'r.-', alpha=0.5)
		# 設定 X 軸位置
		ax.set_xlim(-1, 97)
		tick_locations = [0] + list(range(3, 97, 4))
		tick_labels =  [1] + list(range(4, 97, 4))
		ax.set_xticks(tick_locations)
		ax.set_xticklabels(tick_labels)

		ax.set_title('label_ ' + str(cluster_label), fontsize=32)
		ax.set_xlabel('period', fontsize=20)
		ax.set_ylabel('w', fontsize=20)
	# 儲存 K-Means n 分群的視覺化矩陣圖
	save_n_clusters_visualization_matrix(n_clusters)
	plt.show()
