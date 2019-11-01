
"""
Could also do `spark-submit ratings-counter.py` in cmd.
"""


from pyspark import SparkConf, SparkContext


conf = SparkConf().setMaster("local").setAppName("RatingHistogram")
sc = SparkContext(conf = conf)

lines = sc.textFile("data")
ratings = lines.map(lambda x: x.split()[2])
result = ratings.countByValue()
print result.items()