#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''

Usage:

	$ spark-submit basic_rec_val.py dirname rank regParam k random_seed

'''

import sys
import time
import datetime

from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALSModel
from pyspark.mllib.evaluation import RankingMetrics
from pyspark.sql.types import ArrayType, IntegerType, StructType
from pyspark.sql import Window
from pyspark.sql.functions import column, expr, collect_set, udf
from pyspark.ml.evaluation import RegressionEvaluator


def extract_item(recommendations):
	items = [item_rating[0] for item_rating in recommendations if item_rating[1] >= 3.0]
	return items

def basic_rec_val(spark, dirname, rank, regParam, k, random_seed):
	
	val_set = spark.read.parquet(f'{dirname}/val.parquet')
	
	print(f'Validating on model with rank = {rank} and regParam = {regParam} trained using {dirname} data ...')
	
	# load corresponding trained model
	model = ALSModel.load(f'{dirname}/{rank}_{regParam}_model')

	# computing RMSE on validation set
	predictions = model.transform(val_set)
	evaluator = RegressionEvaluator(metricName='rmse', labelCol='rating', predictionCol='prediction')
	rmse = evaluator.evaluate(predictions)

	print(f'rmse: {rmse}')
							
	print(f'Constructing top {k} books recommended to per user ...')
	val_users = val_set.select('user_id').distinct()

	start_time = time.time()

	perUserPredictedTopKItemsDF = model.recommendForUserSubset(val_users, k)

	myudf = udf(extract_item, ArrayType(IntegerType()))
	perUserPredictedTopKItemsDF = perUserPredictedTopKItemsDF.withColumn('predictions', myudf(perUserPredictedTopKItemsDF['recommendations'])).drop('recommendations')

	print('Constructing actual books per user ...')
	perUserActualItemsDF = val_set.filter(column('rating') >= 3.0).groupBy('user_id').agg(expr('collect_list(book_id) as book_ids'))

	print('Constructing Ranking Metrics ...')
	perUserItemsRDD = perUserPredictedTopKItemsDF.join(perUserActualItemsDF, 'user_id').rdd.map(lambda row: (row[1], row[2]))

	rankingMetrics = RankingMetrics(perUserItemsRDD)

	precisionAtK = rankingMetrics.precisionAt(k)
	mAP = rankingMetrics.meanAveragePrecision

	end_time = time.time()
	time_delta = str(datetime.timedelta(seconds = end_time - start_time))

	print(f'p@{k}: {precisionAtK}')
	print(f'mAP: {mAP}')
	print(f'run time: {time_delta}')

if __name__ == '__main__':

	spark = SparkSession \
			.builder \
			.appName('basic_rec_val') \
			.master('yarn') \
			.config('spark.executor.memory', '8g') \
			.config('spark.driver.memory', '8g') \
			.getOrCreate()
	
	dirname = sys.argv[1]
	rank = int(sys.argv[2])
	regParam = float(sys.argv[3])
	k = int(sys.argv[4])
	random_seed = int(sys.argv[5])

	basic_rec_val(spark, dirname, rank, regParam, k, random_seed)
