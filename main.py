import numpy as np
import pandas as pd

from lib.preprocess import dataset_preprocessing

def main():
	# 資料前置處理
	preprocess_dataSet = dataset_preprocessing.start()
	print('End')

if __name__ == '__main__':
	main()