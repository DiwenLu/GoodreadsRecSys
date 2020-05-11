#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Use Spark's alternating least squares (ALS) method to learn latent factor representations for users and items.
We use the Goodreads dataset collected by 
Mengting Wan, Julian McAuley, "Item Recommendation on Monotonic Behavior Chains", RecSys 2018.

Usage:

	$ spark-submit basic_rec_train.py dirname rank regParam random_seed

'''

import sys
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS

MAXITER = 10

def basic_rec_train(spark, dirname, rank, regParam, random_seed):
	
	# Read training data	
	train = spark.read.parquet(f'{dirname}/train.parquet')
	
	# Start training
	# we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics
	als = ALS(rank=rank, maxIter=MAXITER, regParam = regParam, seed=random_seed, coldStartStrategy='drop', userCol='user_id', itemCol='book_id', ratingCol='rating')
	model = als.fit(train)

	# Save the model
	model.write().overwrite().save(f'{dirname}/{rank}_{regParam}_model')

if __name__ == '__main__':
	 
	spark = SparkSession \
			.builder \
			.appName('basic_rec_train') \
			.master('yarn') \
			.config('spark.executor.memory', '5g') \
			.config('spark.driver.memory', '5g') \
			.getOrCreate()
	
	dirname = sys.argv[1]
	rank = int(sys.argv[2])
	regParam = float(sys.argv[3])
	random_seed = int(sys.argv[4])

	basic_rec_train(spark, dirname, rank, regParam, random_seed)
