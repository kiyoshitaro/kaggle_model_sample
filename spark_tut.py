numerical_cols = ['MSSubClass',
 'LotFrontage',
 'LotArea',
 'OverallQual',
 'OverallCond',
 'YearBuilt',
 'YearRemodAdd',
 'MasVnrArea',
 'BsmtFinSF1',
 'BsmtFinSF2',
 'BsmtUnfSF',
 'TotalBsmtSF',
 '1stFlrSF',
 '2ndFlrSF',
 'LowQualFinSF',
 'GrLivArea',
 'BsmtFullBath',
 'BsmtHalfBath',
 'FullBath',
 'HalfBath',
 'BedroomAbvGr',
 'KitchenAbvGr',
 'TotRmsAbvGrd',
 'Fireplaces',
 'GarageYrBlt',
 'GarageCars',
 'GarageArea',
 'WoodDeckSF',
 'OpenPorchSF',
 'EnclosedPorch',
 '3SsnPorch',
 'ScreenPorch',
 'MiscVal',
 'MoSold',
 'YrSold']


nominal = ['MSZoning',
 'LotShape',
 'LandContour',
 'LotConfig',
 'LandSlope',
 'Neighborhood',
 'Condition1',
 'Condition2',
 'BldgType',
 'HouseStyle',
 'RoofStyle',
 'RoofMatl',
 'Exterior1st',
 'Exterior2nd',
 'MasVnrType',
 'Foundation',
 'Heating',
 'Electrical',
 'GarageType',
 'GarageFinish',
 'PavedDrive',
 'SaleType',
 'SaleCondition']

ordinal = ['HeatingQC',
 'ExterQual',
 'GarageCond',
 'BsmtQual',
 'CentralAir',
 'Functional',
 'GarageQual',
 'BsmtFinType1',
 'BsmtCond',
 'BsmtExposure',
 'FireplaceQu',
 'ExterCond',
 'KitchenQual',
 'BsmtFinType2',
 'TotalGarageQual',
 'TotalExteriorQual']

# CREATE SPARK CORE (define RDD)
from pyspark import SparkContext
sc = SparkContext.getOrCreate()
from pyspark.sql import SparkSession
my_spark = SparkSession.builder.getOrCreate()
my_spark.catalog.listTables()
train_df = my_spark.read.csv('train.csv', header = True)
y = train_df.SalePrice
train_df.drop(numerical_cols).columns
test_df = my_spark.read.csv('houseprice/test.csv', header = True)
idx = test_df["Id"]
df = pd.concat([train_df, test_df],ignore_index=True)
df.drop("Id").columns

miss_feature = ["Alley","MiscFeature","PoolQC","Fence"]
df.drop(columns=miss_feature).columns

for i in numerical_cols:
    df = df.withColumn(i, df.i.cast("integer"))

from pyspark.ml.feature import StringIndexer, OneHotEncoder

stages = []
inp_cols = numerical_cols
for i in nominal:
    stages.append(StringIndexer(inputCol = i, \
                                outputCol = i + "_index"))
    stages.append(OneHotEncoder(inputCol =  i + "_index", \
                                outputCol = i + "_fact"))
    inp_cols.append(i + "_fact")

from pyspark.ml.feature import VectorAssembler
vec_assembler = VectorAssembler(inputCols = inp_cols, 
                                outputCol = "features")
stages.append(vec_assembler)

from pyspark.ml import Pipeline
# Make a pipeline
houseprice_pipe  = Pipeline(stages = stages)
pipe_data = houseprice_pipe.fit(df).transform(df)
train, test = pipe_data.randomSplit([0.8, 0.2])

from pyspark.ml.regression import LinearRegression
lr = LinearRegression()
import pyspark.ml.evaluation as evals
evaluator = evals.RegressionEvaluator(metricName = "rmse")
# Import the tuning submodule
import pyspark.ml.tuning as tune
import numpy as np

grid = tune.ParamGridBuilder()
grid = grid.addGrid(lr.regParam, np.arange(0, 0.1, 0.01))
grid = grid.build()
cv = tune.CrossValidator(estimator=lr,
               estimatorParamMaps=grid,
               evaluator=evaluator
               )

# Fit cross validation on models
models = cv.fit(train)
best_lr = models.bestModel
best_lr = lr.fit(train)
test_results = best_lr.transform(test)
print(evaluator.evaluate(test_results))