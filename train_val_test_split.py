#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Usage:

	$ spark-submit train_val_test_split.py dirname random_seed
'''

import sys
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import ArrayType, IntegerType
from pyspark.sql.functions import column, row_number
from pyspark.sql import Window

TRAIN_FRAC = 0.6
VAL_FRAC = 0.2
TEST_FRAC = 0.2

def train_val_test_split(spark, dirname, random_seed):

	# Read in downsampled interactions
	interactions = spark.read.parquet(f'{dirname}/{dirname}_subsamples.parquet')

	interactions = interactions.filter(interactions.rating > 0).drop('is_read', 'is_reviewed')

	# Find all unique user ids with interactions number > 10
	userids = interactions \
					.groupby('user_id') \
					.count().alias('count') \
					.filter(column('count') > 10) \
					.select('user_id')

	# Sample train, val, test user
	train_user = userids.sample(False, TRAIN_FRAC, rand_seed)
	remaining_user = userids.subtract(train_user)
	val_user = remaining_user.sample(False, VAL_FRAC / (1 - TRAIN_FRAC), rand_seed)
	test_user = remaining_user.subtract(val_user)

	# Construct train, val, test interactions
	train_all = train_user.join(interactions, on = 'user_id', how = 'inner')
	val_all = val_user.join(interactions, on = 'user_id', how = 'inner')
	test_all = test_user.join(interactions, on = 'user_id', how = 'inner')

	# Split val and test in half
	window = Window.partitionBy('user_id').orderBy('book_id')

	val_interactions = (val_all.select("user_id","book_id","rating", row_number().over(window).alias("row_number")))
	test_interactions = (test_all.select("user_id","book_id","rating", row_number().over(window).alias("row_number")))

	val_add2train = val_interactions.filter(val_interactions.row_number % 2 == 0).drop('row_number')
	val = val_interactions.filter(val_interactions.row_number % 2 == 1).drop('row_number')

	test_add2train = test_interactions.filter(test_interactions.row_number % 2 == 1).drop('row_number')
	test = test_interactions.filter(test_interactions.row_number % 2 == 0).drop('row_number')

	# Add half val and half test back to train
	train = train_all.union(val_add2train).union(test_add2train)

	# Write train_set, val_set, test_set out
	train.write.mode('overwrite').parquet(f'{dirname}/train.parquet')
	val.write.mode('overwrite').parquet(f'{dirname}/val.parquet')
	test.write.mode('overwrite').parquet(f'{dirname}/test.parquet')

if __name__ == '__main__':

	spark = SparkSession.builder.appName('basic_rec_train').getOrCreate()

	filename = sys.argv[1]
	random_seed = int(sys.argv[2])

	train_val_test_split(spark, dirname, random_seed)
