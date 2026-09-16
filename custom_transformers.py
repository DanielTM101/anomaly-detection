import base64
import numpy as np
from pyspark.ml import Transformer, Estimator, Model
from pyspark.ml.param.shared import Param, Params
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable
from pyspark.sql.functions import col, when, abs as abs_, lit, hour


# 1) Domain transformer for Energy data


class MultiDomainRuleTransformer(Transformer, DefaultParamsReadable, DefaultParamsWritable):
    """
    Adds domain-based flags for Energy data:
      - timeFlag
      - voltageFlag
      - solarFlag
      - freqFlag
    """
    def __init__(self, usageThreshold=70, freqThreshold=0.5, **kwargs):
        super().__init__(**kwargs)
        self.usageThreshold = usageThreshold
        self.freqThreshold = freqThreshold

    def _transform(self, dataset):
        df2 = dataset.withColumn(
            "timeFlag",
            when(
                (
                    ((col("hour") >= 10) & (col("hour") < 15))
                    | (col("hour") >= 22)
                    | (col("hour") < 5)
                ) & (col("energy_consumption") > self.usageThreshold),
                1
            ).otherwise(0)
        )
        df2 = df2.withColumn(
            "voltageFlag",
            when((col("voltage") < 180) | (col("voltage") > 260), 1).otherwise(0)
        )
        df2 = df2.withColumn(
            "solarFlag",
            when(
                (((col("hour") >= 6) & (col("hour") < 18) & (col("solar_output") == 0))
                 | (col("solar_output") > 30)),
                1
            ).otherwise(0)
        )
        df2 = df2.withColumn(
            "freqFlag",
            when(abs_(col("frequency") - 50) > self.freqThreshold, 1).otherwise(0)
        )
        return df2


# 2) Domain transformer for Network data


class NetworkDomainRuleTransformer(Transformer, DefaultParamsReadable, DefaultParamsWritable):
    """
    Adds domain-based flags for Network data:
      - bigPacketFlag
      - synFlag
      - suspiciousPortFlag
      - protocolFlag
    """
    def __init__(self, bigPacketThreshold=1000, **kwargs):
        super().__init__(**kwargs)
        self.bigPacketThreshold = bigPacketThreshold

    def _transform(self, dataset):
        df2 = dataset.withColumn(
            "bigPacketFlag",
            when(col("length") > self.bigPacketThreshold, 1).otherwise(0)
        )
        df2 = df2.withColumn(
            "synFlag",
            when(col("syn") == 1, 1).otherwise(0)
        )
        df2 = df2.withColumn(
            "suspiciousPortFlag",
            when((col("Source Port") < 1024) | (col("Destination Port") < 1024), 1).otherwise(0)
        )
        df2 = df2.withColumn(
            "protocolFlag",
            when((col("Protocol") == "ICMP") | (col("Protocol") == "AMQP"), 1).otherwise(0)
        )
        return df2


# 3) Custom Logistic Regression


from pyspark.ml.feature import VectorAssembler, StandardScaler

class CustomLogisticRegression(Estimator, DefaultParamsReadable, DefaultParamsWritable):
    """
    A toy logistic regression Estimator that uses gradient descent on a collected DataFrame.
    Produces a CustomLogisticRegressionModel containing the learned weights.
    """
    def __init__(self, labelCol="indexedLabel", featuresCol="features", maxIter=20, lr=0.01, **kwargs):
        super().__init__(**kwargs)
        self.labelCol = labelCol
        self.featuresCol = featuresCol
        self.maxIter = maxIter
        self.lr = lr

    def _fit(self, dataset):
        data = dataset.select(self.featuresCol, self.labelCol).collect()
        X_list = []
        y_list = []
        for row in data:
            X_list.append(row[self.featuresCol])
            y_list.append(row[self.labelCol])

        X = np.array(X_list)
        y = np.array(y_list)
        n_samples, n_features = X.shape

        weights = np.zeros(n_features)
        bias = 0.0

        def sigmoid(z):
            return 1.0 / (1.0 + np.exp(-z))

        for _ in range(self.maxIter):
            z = np.dot(X, weights) + bias
            preds = sigmoid(z)
            error = preds - y
            grad_w = np.dot(X.T, error) / n_samples
            grad_b = np.mean(error)
            weights -= self.lr * grad_w
            bias -= self.lr * grad_b

        return CustomLogisticRegressionModel(
            weights=weights,
            bias=bias,
            labelCol=self.labelCol,
            featuresCol=self.featuresCol
        )

class CustomLogisticRegressionModel(Model, DefaultParamsReadable, DefaultParamsWritable):
    """
    The fitted model containing the learned weights and bias. 
    Implements a no-arg constructor so Spark can load it from disk.
    We store weights/bias in a JSON-serializable form.
    """

    # For param-based storage:
    weightsParam = Param(Params._dummy(), "weightsParam", "numpy weights array serialized (base64 encoded)")
    biasParam = Param(Params._dummy(), "biasParam", "bias float")
    labelColParam = Param(Params._dummy(), "labelColParam", "label column name")
    featuresColParam = Param(Params._dummy(), "featuresColParam", "features column name")

    def __init__(self, weights=None, bias=None, labelCol=None, featuresCol=None, **kwargs):
        super().__init__(**kwargs)
        if weights is not None:
            # Encode weights as a base64 string for JSON serialization
            self._setDefault(weightsParam=base64.b64encode(weights).decode('utf-8'))
        if bias is not None:
            self._setDefault(biasParam=float(bias))
        if labelCol is not None:
            self._setDefault(labelColParam=labelCol)
        if featuresCol is not None:
            self._setDefault(featuresColParam=featuresCol)

    def _transform(self, dataset):
        from pyspark.sql.functions import udf, when, col
        from pyspark.sql.types import DoubleType
        import numpy as np

        # Decode the weights from the base64 string
        raw_weights_str = self.getOrDefault(self.weightsParam)
        weights_array = np.frombuffer(base64.b64decode(raw_weights_str), dtype=np.float64)
        bias_float = self.getOrDefault(self.biasParam)

        def predict_prob(feature_vec):
            arr = np.array(feature_vec)
            z = np.dot(arr, weights_array) + bias_float
            return float(1.0 / (1.0 + np.exp(-z)))

        predict_udf = udf(predict_prob, DoubleType())

        df_with_prob = dataset.withColumn(
            "probability_custom",
            predict_udf(col(self.getOrDefault(self.featuresColParam)))
        )
        df_final = df_with_prob.withColumn(
        "prediction",
        when(col("probability_custom") >= 0.3, 1.0).otherwise(0.0)
	)
        return df_final

    def _copyValues(self, to, extra=None):
        """Copy param values from self to 'to' (another instance)."""
        super()._copyValues(to, extra)
        if self.isSet(self.weightsParam):
            to.set(to.weightsParam, self.getOrDefault(self.weightsParam))
        if self.isSet(self.biasParam):
            to.set(to.biasParam, self.getOrDefault(self.biasParam))
        if self.isSet(self.labelColParam):
            to.set(to.labelColParam, self.getOrDefault(self.labelColParam))
        if self.isSet(self.featuresColParam):
            to.set(to.featuresColParam, self.getOrDefault(self.featuresColParam))
        return to

    def copy(self, extra=None):
        newModel = CustomLogisticRegressionModel()
        return self._copyValues(newModel, extra=extra)
