#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Usage: 

	$ spark-submit annoy_train.py dirname rank regParam n_trees random_seed

Code reference: https://github.com/spotify/annoy
'''

import sys
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALSModel
from annoy import AnnoyIndex
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number
from tqdm import tqdm

def convert_annoy_index(item_factors):
    window = Window.orderBy('id')
    item_factors = item_factors.withColumn('annoy_id', row_number().over(window))
    annoy_index_map = item_factors.select('id', 'annoy_id')
    item_factors = item_factors.select('annoy_id', 'features')
    return item_factors, annoy_index_map

def annoy_train(spark, dirname, rank, regParam, n_trees, random_seed):

	# Load model
	model = ALSModel.load(f'{dirname}/{rank}_{regParam}_model')
	
	# get item factors
	item_factors = model.itemFactors
	item_factors, annoy_index_map = convert_annoy_index(item_factors)
	
	# train annoy model
	tree = AnnoyIndex(rank, 'dot')
	for item in tqdm(item_factors.collect()):
		tree.add_item(item.annoy_id, item.features)
	tree.set_seed(random_seed)
	
	# build the tree
	# num of trees: higher n_trees gives higher precision
	tree.build(n_trees)
	
	# save annoy model and index map
	tree.save(f'{dirname}_{rank}_{regParam}_tree.ann')
	annoy_index_map.write.parquet(f'{dirname}_{rank}_{regParam}_annoy_index_map.parquet')

if __name__ == '__main__':
	spark = SparkSession.builder \
						.appName('annoy_train') \
						.master('yarn') \
						.config('spark.executor.memory', '5g') \
						.config('spark.driver.memory', '5g') \
						.getOrCreate()

	dirname = sys.argv[1]
	rank = int(sys.argv[2])
	regParam = float(sys.argv[3])
	n_trees = int(sys.argv[4])
	random_seed = int(sys.argv[5])

	annoy_train(spark, dirname, rank, regParam, n_trees, random_seed)
