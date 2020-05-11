#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Usage:

	$ spark-submit read_from_source.py
'''

import sys
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, IntegerType, StructField


def read_csv_from(spark, filename):

	schema = StructType([StructField('user_id', IntegerType()), \
						StructField('book_id', IntegerType()), \
						StructField('is_read', IntegerType()), \
						StructField('rating', IntegerType()), \
						StructField('is_reviewed', IntegerType())])

	data = spark.read.csv(filename, header = True, schema = schema)
	data.write.mode('overwrite').parquet('full_interactions.parquet')

if __name__ == '__main__':
	spark = SparkSession \
				.builder \
				.appName('read_from_source') \
				.getOrCreate()

	filename = 'hdfs:/user/bm106/pub/goodreads/goodreads_interactions.csv'
	read_csv_from(spark, filename)
