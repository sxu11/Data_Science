"""
FormattedRegressedConditionDetails
| where property in ("NodeState","*NodeBugCheckFromRdAgent","*DirtyShutDownFromRdAgent","HostAgentVmStateV2","*DcmUnexpectedReboot","*BugCheck","NodeBeingRepaired","*DirtyShutDown","*BmcUpdate")
| extend propertyValue = iff(property =="NodeState",strcat(property,"__",value),property)
| summarize dcount(targetId) by propertyValue, clusterName
"""

import pandas as pd
from pyspark.ml.recommendation import ALS
from pyspark.sql import SparkSession



spark = SparkSession.builder.appName('Recommendation_system').getOrCreate()

df = pd.read_csv("query_results.csv")

clusterNames = df["clusterName"].unique().tolist()
clusterToIdMapper = dict(zip(clusterNames, range(len(clusterNames))))
df["clusterName"] = df["clusterName"].apply(lambda x: clusterToIdMapper[x])

propertyValues = df["propertyValue"].unique().tolist()
propertyValueToIdMapper = dict(zip(propertyValues, range(len(propertyValues))))
df["propertyValue"] = df["propertyValue"].apply(lambda x: propertyValueToIdMapper[x])

print("len(clusterToIdMapper)", len(clusterToIdMapper))
print("len(propertyValueToIdMapper)", len(propertyValueToIdMapper))

als = ALS(maxIter=5,
          regParam=0.1,
          userCol="propertyValue",
          itemCol="clusterName",
          ratingCol="dcount_targetId",
          coldStartStrategy="drop")


sparkDF=spark.createDataFrame(df)

sparkDF.show(10)
model=als.fit(sparkDF)
model.itemFactors.show(10, truncate=False)
