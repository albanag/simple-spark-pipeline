from pyspark.ml.param.shared import HasInputCol, HasOutputCol
from pyspark.sql.functions import udf
from pyspark.ml.pipeline import Transformer
from pyspark import keyword_only


class TemperatureDiscretizer(Transformer, HasInputCol, HasOutputCol):

# todo write some documentation

    @keyword_only
    def __init__(self, inputCol=None, outputCol=None):
        super(TemperatureDiscretizer, self).__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCol=None, outputCol=None):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _transform(self, dataset):
        discretize_temp_udf = udf(lambda s: round(float(str(s).replace(',', '.'))))
        res_df = dataset.withColumn(self.getOutputCol(), discretize_temp_udf(dataset[self.getInputCol()]))
        return res_df
