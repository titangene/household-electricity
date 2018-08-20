import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def load_dataset(data_path, dtype=None, parse_dates=None):
	dataframe = pd.read_csv('data/' + data_path, dtype=dtype, parse_dates=parse_dates)
	return dataframe

def save_csv(data, data_path):
	data.to_csv('data/' + data_path, encoding='utf-8', index=False)
	print('save: ' + data_path)

# 遇到 負數 直接砍，因為發現 sensor 本身有問題
# 刪除 >= 10000
def delete_outliers(dataeSet, threshold=10000):
	return dataeSet[(dataeSet['w'] >= 0) & (dataeSet['w'] <= threshold)]

# string to datetime
def transform_time(dataeSet, column, format):
	# dataeSet.loc[:, 'reporttime'] = pd.to_datetime(dataeSet['reporttime'])
	# dataeSet['reporttime'] = pd.to_datetime(dataeSet['reporttime'], format='%Y-%M-%d')
	dataeSet[column] = pd.to_datetime(dataeSet[column], format=format)
	return dataeSet

# 以 buildingid 分類，彙整每個使用者用電資料為每 15 分鐘一筆，w 四捨五入至小數 2 位
def groupbyData(dataSet):
	group_dataSet = dataSet.groupby(['buildingid', pd.Grouper(key='reporttime', freq='15T')])['w'].mean().round(2).reset_index()
	return group_dataSet

# 建立以日為單位之欄位 (96 期)
def create_peroid_column():
	return ['period_' + str(idx) for idx in range(1, 97)]

# 建立彙整資料欄位
def create_consolidation_column():
	return ['uuid', 'buildingId', 'reportTime'] + create_peroid_column()

# 建立新的 period 時間 list
def create_periods_datetime_list():
	return pd.date_range('00:00:00', periods=96, freq='15T').time

# 轉置用電資料
def transpose_data_electricity_watt(date_df):
	period_index = 0
	df_list = []
	if (len(date_df) == 96):
		return date_df.drop(['reporttime'], axis=1)['w'].tolist()
	else:
		for index, row in date_df.iterrows():
			periods = create_periods_datetime_list()
			# 直到找到該時段的 index
			while (row['reporttime'].time() != periods[period_index]):
				df_list.append(None)
				period_index += 1

			# print(period_index, row['reporttime'].time(), periods[period_index], row['reporttime'].time() == periods[period_index])
			df_list.append(row['w'])
			period_index += 1

		# 將最後面的 NA 值設為 None
		if (len(df_list) != 96):
			df_list.append(None)

			while (len(df_list) != 96):
				df_list.append(None)
				period_index += 1
	return df_list

# 建立一天的資料集
def set_day_dataSet(uuid, buildingId, reportTime, date_df):
	# print(" ", reportTime, len(date_df))
	data_watt_list = transpose_data_electricity_watt(date_df)

	# [uuid, buildingId, reportTime, period_1...]
	dataSet_list = [uuid, buildingId, reportTime] + data_watt_list

	# if (len(dataSet_list) != 99):
	# 	print(dataSet_list[2], len(dataSet_list))
	return dataSet_list

# 彙整與轉置單一使用者的用電資料 (96 期)
def consolidation_buildingid_dataSet(date_groups, user_groupName):
	dataSet_lists = []

	for date_groupName, date_group in date_groups:
		date_df = date_group.reset_index()
		date_df = date_df.drop(['index', 'buildingid'], axis=1)

		buildingId = user_groupName
		reportTime = date_groupName.strftime('%Y%m%d')
		channelId = 0
		uuid = '{}{}{}'.format(buildingId, channelId, reportTime)

		reportTime = date_groupName.date()
		dataSet_list = set_day_dataSet(uuid, buildingId, reportTime, date_df)
		dataSet_lists.append(dataSet_list)

	return dataSet_lists

# 彙整與轉置多個使用者的用電資料 (96 期)
def consolidation_all_dataSet(dataSet):
	users_group = dataSet.groupby('buildingid')
	users_dataSet_list = []

	for user_groupName, user_group in users_group:
		date_groups = user_group.groupby(pd.Grouper(key='reporttime', freq='1D'))
		tmp_list = consolidation_buildingid_dataSet(date_groups, user_groupName)
		users_dataSet_list += tmp_list
		print('process buildingid{}\t{}'.format(user_groupName, len(users_dataSet_list)))

	return users_dataSet_list

# 刪除最前或最後有缺值之資料
def delete_first_or_last_na(dataSet):
	# tmp_dataSet = dataSet[dataSet['period_1'].notnull()]
	# return tmp_dataSet[tmp_dataSet['period_96'].notnull()]
	return dataSet.dropna(subset=['period_1', 'period_96'])

# 刪除缺值之門檻值
def dorpna_threshold(dataSet, threshold):
	period_sum = 96
	# uuid, buildingId, reportTime
	another_column_sum = 3
	# return dataSet.dropna(thresh=(period_sum - threshold + another_column_sum))
	return dataSet.dropna(thresh=(11 - threshold + 1))

# 將 period_10 轉成 10
def transforma_period_list_number(na_periods_colume):
	return [int(period.replace('period_', '')) for period in na_periods_colume]

# period 補值
def fill_period_na(row, na_periods, current_idx):
	current_period = na_periods['colume'][current_idx]
	na_periods['current'] = na_periods['list'][current_idx]
	prev_period = 'period_' + str(na_periods['current'] - 1)
	next_period = 'period_' + str(na_periods['current'] + 1)

	period_ave = (row[next_period] + row[prev_period]) / 2

	# print('{}: {}, p{}_{}, n{}_{} = c{}-{}'.format(
	# 	row['uuid'], 'fill', 
	# 	(na_periods['current'] - 1), row[prev_period],
	# 	(na_periods['current'] + 1), row[next_period],
	# 	na_periods['current'], round(period_ave, 2)))

	row[current_period] = round(period_ave, 2)

# period 缺值處理
def process_period_na(row):
	# print('\n' + '=' * 40)
	na_periods = {}
	na_periods['colume'] = row.index[row.isnull()].tolist() 
	na_periods['len'] = len(na_periods['colume'])

	if (na_periods['len'] == 0):
		# print('o', row['uuid'], na_periods['len'])
		return row
	else:
		na_periods['list'] = transforma_period_list_number(na_periods['colume'])
		# print('x', row['uuid'], na_periods['len'], na_periods['colume'])

		if (na_periods['len'] == 1):
			fill_period_na(row, na_periods, 0)
		else:
			for idx in range(na_periods['len'] - 1):
				na_periods['current'] = na_periods['list'][idx]
				na_periods['next'] = na_periods['list'][idx + 1]
				# print(na_periods['current'], na_periods['next'])

				if (na_periods['next'] - na_periods['current'] == 1):
					row[row['uuid'] != row['uuid']]
					# print(row['uuid'], 'drop', na_periods['list'])
					# print('drop row', row['uuid'], row.name)
					return np.nan
				else:
					fill_period_na(row, na_periods, idx)

				# print('-' * 40)
			# print(na_periods['list'][na_periods['len'] - 1])
# 			fill_period_na(row, na_periods, na_periods['len'] - 1)
# 	print('fill row', row['uuid'], type(row['uuid']), row.name)
	return row

def process_na(dataSet, peroid_column, threshold):
	delete_before_count = len(dataSet)
	dataSet = dorpna_threshold(dataSet, threshold=threshold)
	print('刪除未達門檻值之資料，before: {}, after: {}'.format(delete_before_count, len(dataSet)))

	delete_before_count = len(dataSet)
	dataSet = delete_first_or_last_na(dataSet)
	print('刪除最前或最後有缺值之資料，before: {}, after: {}'.format(delete_before_count, len(dataSet)))

	delete_before_count = len(dataSet)
	dataSet = dataSet.apply(process_period_na, axis=1)
	dataSet = dataSet.dropna(how='all')
	
	# 轉型別
	dataSet['uuid'] = dataSet['uuid'].astype(np.int64).astype(str)
	dataSet['buildingId'] = dataSet['buildingId'].astype(int)
	dataSet = transform_time(dataSet, column='reportTime', format='%Y-%m-%d')

	print('刪除無法補值之資料，before: {}, after: {}'.format(delete_before_count, len(dataSet)))
	return dataSet

def group(sequence, chunk_size):
	return list(zip(*[iter(sequence)] * chunk_size))

# 計算 每小時 (4 個 period) 平均用電
def hour_mean_w(period_group, dataSet):
	tmp_dataSet = dataSet.loc[:, period_group]
	tmp_list = tmp_dataSet.agg('mean', axis=1)
	return tmp_list

# 計算 最大需量 和 最大需量
def peroid_max_min_sum_w(dataSet):
	peroid_column = create_peroid_column()
	# 只取 period 欄位計算 max 和 min
	tmp_dataSet = dataSet.loc[:, peroid_column]
	dataSet['wMax'] = tmp_dataSet.agg('max', axis=1)
	dataSet['wMin'] = tmp_dataSet.agg('min', axis=1)

	period_groups = group(peroid_column, 4)
	tmp_mean_dataSet = pd.DataFrame(list(map(lambda x: hour_mean_w(x, tmp_dataSet), period_groups))).T
	dataSet['wSum'] = tmp_mean_dataSet.agg('sum', axis=1)

	return dataSet.round({'wMax': 2, 'wMin': 2, 'wSum': 2})

# 過濾 buildingId
def set_mask_buildingId(dataSet, buildingId):
	if (buildingId != None):
		# 只有一個用戶
		if (isinstance(buildingId, int)):
			return dataSet['buildingId'] == buildingId
		# 多個用戶
		elif (isinstance(buildingId, list)):
			return dataSet['buildingId'].isin(buildingId)

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
def mask_dataSet(dataSet, buildingId, startTime, endTime):
	mask_buildingId = set_mask_buildingId(dataSet, buildingId)
	mask_time = set_mask_time(dataSet, startTime, endTime)

	if ((is_Series(mask_buildingId)) & (is_Series(mask_time))):
		return dataSet.loc[mask_buildingId & mask_time]
	elif (is_Series(mask_buildingId)):
		return dataSet.loc[mask_buildingId]
	elif (is_Series(mask_time)):
		return dataSet.loc[mask_time]
	else:
		return dataSet

def visualization_single(dataSet, buildingId):
	peroid_column = m.create_peroid_column()
	dataSet = dataSet.loc[:, peroid_column].T

	# grid=False
	# kind='scatter'
	dataSet.plot(colormap='rainbow', alpha=0.5, legend=False, figsize=(20, 8))

	tick_locations = [0] + list(range(3, 97, 4))
	tick_labels =  [1] + list(range(4, 97, 4))
	xmin, xmax, ymin, ymax = plt.axis()
	plt.axis([-1, 97, ymin, ymax])
	plt.xticks(tick_locations, tick_labels)

	plt.title('data visualization\nbuildingId ' + str(buildingId), fontsize=32)
	plt.xlabel('period', fontsize=20)
	plt.ylabel('w', fontsize=20)

def visualization_matrix(dataSet, buildingId_len, ncols):
	grouped = dataSet.groupby('buildingId')
	peroid_column = m.create_peroid_column()

	if (buildingId_len % ncols == 0):
		nrows = buildingId_len // ncols
	else:
		nrows = (buildingId_len // ncols) + 1

	fig, axes = plt.subplots(figsize=(20 * ncols, 6 * nrows),
							 nrows=nrows, ncols=ncols,
							 gridspec_kw=dict(hspace=0.5, wspace=0.12))

	targets = zip(grouped.groups.keys(), axes.flatten())
	for i, (groupName, ax) in enumerate(targets):
		ax.plot(grouped.get_group(groupName).loc[:, peroid_column].T, alpha=0.5)
		# 設定 X 軸位置
		ax.set_xlim(-1, 97)
		tick_locations = [0] + list(range(3, 97, 4))
		tick_labels =  [1] + list(range(4, 97, 4))
		ax.set_xticks(tick_locations)
		ax.set_xticklabels(tick_labels)

		ax.set_title('buildingId ' + str(groupName), fontsize=32)
		ax.set_xlabel('period', fontsize=20)
		ax.set_ylabel('w', fontsize=20)
	ax.legend()
	plt.savefig('visualization.svg', dpi=150)
	plt.show()

#	 nrows = (buildingId_len // ncols + 1, buildingId_len // ncols)[buildingId_len % ncols == 0]
#	 fig, axes = plt.subplots(nrows=nrows, ncols=ncols)

#	 for idx, (groupName, group) in enumerate(grouped):
#		 ax_idx = idx // ncols
#		 ax_idy = idx % ncols

#		 if (nrows == 1):
#			 ax = axes[ax_idy]
#		 else:
#			 ax = axes[ax_idx, ax_idy]

#		 tmp_dataSet = group.loc[:, peroid_column].T
#		 plot(tmp_dataSet, ax=ax)

#	 plt.subplots_adjust(top=0.98, bottom=0.02, left=0.02, right=0.98, hspace=0.6, wspace=0.1)
#	 plt.subplots_adjust(top=0.8, left=0.02, right=0.98, wspace=0.1, hspace=0.8)
#	 plt.tight_layout()
#	 plt.suptitle('data visualization', fontsize=40)

# 資料視覺化
def visualization(dataSet, buildingId=None, startTime=None, endTime=None, drawMode=None, ncols=5):
	dataSet = mask_dataSet(dataSet, buildingId, startTime, endTime)

	# 多個用戶
	if ((buildingId == None) | (isinstance(buildingId, list))):
		if (buildingId == None):
			buildingId_len = len(dataSet['buildingId'].unique())
		else:
			buildingId_len = len(buildingId)

		visualization_matrix(dataSet, buildingId_len, ncols=ncols)
	# 只有一個用戶
	elif (isinstance(buildingId, int)):
		visualization_single(dataSet, buildingId)

def main():
	# 讀取原始 .csv 檔
	dataSet = load_dataset('table_0502.csv')

	# 先做 channelId 0
	channelId_0_dataSet = dataSet[dataSet['channelid'] == 0]
	save_csv(channelId_0_dataSet, '1_channelId_0_dataSet.csv')

	# 遇到 負數 直接砍，因為發現 sensor 本身有問題
	delete_outliers_dataSet = delete_outliers(channelId_0_dataSet)

	# 改變 'reporttime' 欄位 type (string to datetime)
	delete_outliers_dataSet = transform_time(delete_outliers_dataSet, column='reporttime', format='%Y-%m-%d %H:%M:%S')
	save_csv(delete_outliers_dataSet, '2_delete_outliers_dataSet.csv')

	# 以 buildingid 分類，彙整每個使用者用電資料為每 15 分鐘一筆，w 四捨五入至小數 2 位
	group_dataSet = groupbyData(transform_dataSet)
	save_csv(group_dataSet, '3_group_dataSet.csv')

	# 建立彙整資料欄位
	peroid_column = create_consolidation_column()

	# 彙整與轉置多個使用者的用電資料 (96 期)
	consolidation_dataSet_list = consolidation_all_dataSet(group_dataSet)
	consolidation_dataSet = pd.DataFrame(consolidation_dataSet_list, columns=peroid_column)
	save_csv(consolidation_dataSet, '4_consolidation_dataSet.csv')

	# 缺值處理
	fillna_dataSet = process_na(consolidation_dataSet, peroid_column, threshold=2)
	save_csv(fillna_dataSet, '5_fillna_dataSet.csv')

	# 計算 最大需量、最大需量、總用電量
	max_min_sum_w_dataSet = peroid_max_min_sum_w(fillna_dataSet)
	save_csv(max_min_sum_w_dataSet, '6_max_min_sum_w_dataSet.csv')

if (__name__ == '__main__'):
	main()