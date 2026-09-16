from pyspark.sql import SparkSession
from pyspark.sql.functions import col, to_timestamp, hour
from pyspark.sql.types import StructType, StructField, StringType, DoubleType
from pyspark.ml.feature import VectorAssembler, StringIndexer, StandardScaler
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

# Import the custom classes
from custom_transformers import (
    MultiDomainRuleTransformer,
    CustomLogisticRegression
)

spark = SparkSession.builder.appName("EnergyCustomLogisticRegTraining").getOrCreate()

energy_schema = StructType([
    StructField("timestamp", StringType(), True),
    StructField("energy_consumption", DoubleType(), True),
    StructField("voltage", DoubleType(), True),
    StructField("current", DoubleType(), True),
    StructField("power", DoubleType(), True),
    StructField("solar_output", DoubleType(), True),
    StructField("frequency", DoubleType(), True),
    StructField("label", StringType(), True)
])

df = spark.read.csv(
    "hdfs://192.168.56.101:9000/home/dan/project/energy_dataset.csv",
    header=True,
    schema=energy_schema
).dropna()

df = df.withColumn("hour", hour(to_timestamp(col("timestamp"), "dd/MM/yyyy HH:mm:ss")))

multiDomain = MultiDomainRuleTransformer(usageThreshold=70, freqThreshold=0.5)
label_indexer = StringIndexer(inputCol="label", outputCol="indexedLabel").setHandleInvalid("keep")


feature_cols = [
    "energy_consumption","voltage","current","power","solar_output","frequency",
    "timeFlag","voltageFlag","solarFlag","freqFlag"
]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="rawFeatures")
scaler = StandardScaler(inputCol="rawFeatures", outputCol="features", withStd=True, withMean=False)

myLR = CustomLogisticRegression(labelCol="indexedLabel", featuresCol="features", maxIter=30, lr=0.01)

pipeline = Pipeline(stages=[multiDomain, label_indexer, assembler, scaler, myLR])

train_df, test_df = df.randomSplit([0.7, 0.3], seed=42)
model = pipeline.fit(train_df)

predictions = model.transform(test_df)
evaluator = MulticlassClassificationEvaluator(labelCol="indexedLabel", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print(f"[EnergyModel] Accuracy: {accuracy:.3f}")

model.write().overwrite().save("hdfs://192.168.56.101:9000/home/dan/project/oop_custom_energy_model")
print("[EnergyModel] Pipeline saved successfully!")
spark.stop()
