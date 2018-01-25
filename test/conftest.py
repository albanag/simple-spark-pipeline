import pytest
from pyspark.sql import SparkSession
import logging


def quiet_py4j():
    """ turn down spark logging for the test context """
    logger = logging.getLogger('py4j')
    logger.setLevel(logging.ERROR)


@pytest.fixture(scope="session")
def spark_context(request):
    """ fixture for creating a spark context
    Args:
        request: pytest.FixtureRequest object
    """
    spark = SparkSession.builder.master("local[1]").appName("pytest-test").getOrCreate()
    request.addfinalizer(lambda: spark.stop())

    quiet_py4j()
    return spark
