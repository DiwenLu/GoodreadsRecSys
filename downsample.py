#!/usr/bin/python
# -*- coding: utf-8 -*-

'''

Usage:

	$ spark-submit downsample.py k random_seed

'''

import sys
from pyspark.sql import SparkSession


def downsample(spark, k, rand_seed, filename):

	# Load entie data set
	full_interactions = spark.read.parquet('full_interactions.parquet')
	full_users = full_interactions.select('user_id').distinct()
	
	# Sample k % users
	k_percent_users = full_users.sample(False, k / 100, rand_seed)
	k_percent_users_interactions = k_percent_users.join(full_interactions, on = 'user_id', how = 'left')
	k_percent_users_interactions.write.mode('overwrite').parquet(f'{k}_percent/{k}_percent_subsamples.parquet')

if __name__ == '__main__':

	spark = SparkSession.builder.appName('downsample').getOrCreate()
	
	k = int(sys.argv[1])
	rand_seed = int(sys.argv[2])
	
	downsample(spark, k, random_seed)
