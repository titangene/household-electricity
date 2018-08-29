import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ..preprocess import user_load_data

# 過濾 userId
def set_mask_userId(dataSet, userId):
	if (userId != None):
		# 只有一個用戶
		if (isinstance(userId, int)):
			return dataSet['userId'] == userId
		# 多個用戶
		elif (isinstance(userId, list)):
			return dataSet['userId'].isin(userId)

# 過濾 reportTime 的 startTime 和 endTime
def set_mask_time(dataSet, startTime, endTime):
	# 包含 startTime 和 endTime
	if ((startTime != None) & (endTime != None)):
		return dataSet["reportTime"].between(startTime, endTime)
	elif (startTime != None):
		return dataSet['reportTime'] >= startTime
	elif (endTime != None):
		return dataSet['reportTime'] <= endTime

# 是否為 Series
def is_Series(series):
	return isinstance(series, pd.core.series.Series)

# 過濾 dataSet
def mask_dataSet(dataSet, userId, startTime, endTime):
	mask_userId = set_mask_userId(dataSet, userId)
	mask_time = set_mask_time(dataSet, startTime, endTime)

	if ((is_Series(mask_userId)) & (is_Series(mask_time))):
		return dataSet.loc[mask_userId & mask_time]
	elif (is_Series(mask_userId)):
		return dataSet.loc[mask_userId]
	elif (is_Series(mask_time)):
		return dataSet.loc[mask_time]
	else:
		return dataSet

def plot_single(dataSet, userId):
	peroid_column = user_load_data.create_peroid_column()
	dataSet = dataSet.loc[:, peroid_column].T

	# grid=False
	# kind='scatter'
	dataSet.plot(colormap='rainbow', alpha=0.5, legend=False, figsize=(20, 8))

	tick_locations = [0] + list(range(3, 97, 4))
	tick_labels =  [1] + list(range(4, 97, 4))
	xmin, xmax, ymin, ymax = plt.axis()
	plt.axis([-1, 97, ymin, ymax])
	plt.xticks(tick_locations, tick_labels)

	plt.title('data visualization\nuserId ' + str(userId), fontsize=32)
	plt.xlabel('period', fontsize=20)
	plt.ylabel('w', fontsize=20)

def plot_matrix(dataSet, userId_len, ncols):
	grouped = dataSet.groupby('userId')
	peroid_column = user_load_data.create_peroid_column()

	if (userId_len % ncols == 0):
		nrows = userId_len // ncols
	else:
		nrows = (userId_len // ncols) + 1

	fig, axes = plt.subplots(nrows, ncols, figsize=(20 * ncols, 6 * nrows),
							 gridspec_kw=dict(hspace=0.5, wspace=0.12))

	targets = zip(grouped.groups.keys(), axes.flatten())
	for idx, (groupName, ax) in enumerate(targets):
		ax.plot(grouped.get_group(groupName).loc[:, peroid_column].T, alpha=0.5)
		# 設定 X 軸位置
		ax.set_xlim(-1, 97)
		tick_locations = [0] + list(range(3, 97, 4))
		tick_labels =  [1] + list(range(4, 97, 4))
		ax.set_xticks(tick_locations)
		ax.set_xticklabels(tick_labels)

		ax.set_title('userId ' + str(groupName), fontsize=32)
		ax.set_xlabel('period', fontsize=20)
		ax.set_ylabel('w', fontsize=20)
	plt.savefig('visualization.svg', dpi=150)
	plt.show()

#	 plt.subplots_adjust(top=0.98, bottom=0.02, left=0.02, right=0.98, hspace=0.6, wspace=0.1)
#	 plt.subplots_adjust(top=0.8, left=0.02, right=0.98, wspace=0.1, hspace=0.8)
#	 plt.tight_layout()
#	 plt.suptitle('data visualization', fontsize=40)

# 資料視覺化
def plot_dataSet(dataSet, userId=None, startTime=None, endTime=None, drawMode=None, ncols=5):
	dataSet = mask_dataSet(dataSet, userId, startTime, endTime)

	# 多個用戶
	if ((userId == None) | (isinstance(userId, list))):
		if (userId == None):
			userId_len = len(dataSet['userId'].unique())
		else:
			userId_len = len(userId)

		plot_matrix(dataSet, userId_len, ncols=ncols)
	# 只有一個用戶
	elif (isinstance(userId, str)):
		plot_single(dataSet, userId)
