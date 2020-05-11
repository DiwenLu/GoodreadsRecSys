#!/usr/bin/env python
# -*- coding: utf-8 -*-
'''

Usage:

	$ spark-submit item_user_factors.py dirname rank regParam

'''

import sys
from pyspark.ml.recommendation import ALSModel
from pyspark.sql import SparkSession

def itemFactors(spark, dirname, rank, regParam):

	# Load corresponding trained model
	model = ALSModel.load(f'{dirname}/{rank}_{regParam}_model')

	# Extract latent factors from trained model
	itemFactors = model.itemFactors
	userFactors = model.userFactors
	
	# Write out
	itemFactors.write.mode('overwrite').parquet(f'{dirname}/itemFactors_{rank}_{regParam}_{dirname}.parquet')
	userFactors.write.mode('overwrite').parquet(f'{dirname}/userFactors_{rank}_{regParam}_{dirname}.parquet')
	
if __name__ == '__main__':
	spark = SparkSession \
				.builder \
				.appName('item_user_factors') \
				.master('yarn') \
				.config('spark.executor.memory', '3g') \
				.config('spark.driver.memory', '3g') \
				.getOrCreate()

	dirname = sys.argv[1]
	rank = int(sys.argv[2])
	regParam = float(sys.argv[3])

	itemFactors(spark, dirname, rank, regParam)
