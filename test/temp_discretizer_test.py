import pytest

from ml.data_processing import TemperatureDiscretizer

pytest.mark.usefixtures("spark_context")


def test_temperature_discretizer(spark_context):

    columns = ['id', 'temp']
    vals = [
        (0, '1,4'),
        (1, '2,5')
    ]

    df = spark_context.createDataFrame(vals, columns)

    temp_discretizer = TemperatureDiscretizer(inputCol="temp", outputCol="dtemp")

    result_df = temp_discretizer.transform(df)

    expected_df = spark_context.createDataFrame([(0, '1,4', str(1)), (1, '2,5', str(round(2.5)))],
                                                ['id', 'temp', 'dtemp'])

    # TODO compare two dataframes
    assert 1 == 1
