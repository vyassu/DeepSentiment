from __future__ import print_function

import sys,re

from pyspark import SparkContext
from operator import add



if __name__ == "__main__":
    # Validates the arguments given
    if len(sys.argv) != 2:
        print("Usage: wordcount <file>", file=sys.stderr)
        exit(-1)
    
    sc = SparkContext(appName="PythonWordCount")
    # Creating RDD
    lines = sc.textFile(sys.argv[1], 1)
    # Caches the RDD to avoid disk access.
    lines.cache()
    # Transforming the RDD and reducing the results at the Master
    counts = lines.flatMap(lambda y: y.split(' ')).map(lambda x: (str(x), 1)).reduceByKey(add)
    # Storing the results into HDFS directory path given below
    counts.saveAsTextFile("Results-WordCount")

    sc.stop()
