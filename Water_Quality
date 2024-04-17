#!/usr/bin/env python
# coding: utf-8

# # Data Preprocessing

# In[1]:


import pyspark
import pandas as pd
import numpy as np
import math
import requests


# In[2]:


from pyspark import SparkContext
from pyspark.sql import SparkSession
from pyspark.sql.types import StructField, StructType, StringType, LongType, IntegerType, FloatType
from pyspark.sql.functions import col, column, when, avg, mean
from pyspark.sql.functions import expr
from pyspark.sql.functions import split
from pyspark.sql.functions import array_contains, array_position
from pyspark.sql.functions import collect_list
from pyspark.sql import Row
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.regression import DecisionTreeRegressor


# In[3]:


ss=SparkSession.builder.appName("DS410 Project").getOrCreate()


# In[4]:


ss.sparkContext.setCheckpointDir("~/scratch")


# In[5]:


df = ss.read.csv("/storage/home/trn5106/work/Final Project/main code/water_quality_large.csv", header=True, inferSchema=True) # Make sure to use the small dataset


# In[6]:


#df.printSchema()


# In[7]:


df2 = df.select("ActivityStartDate","MonitoringLocationIdentifier","CharacteristicName", "ResultMeasureValue", "HydrologicEvent", "ResultMeasureUnits")


# In[ ]:


#df2.printSchema()


# In[ ]:


#df2.show(5)


# In[ ]:


df3 = df2.filter(col("ActivityStartDate").isNotNull())
df4 = df3.filter(col("MonitoringLocationIdentifier").isNotNull())


# In[ ]:


#df4.filter(col("CharacteristicName") == "Temperature, water").select("ResultMeasure/MeasureUnitCode").summary().show()


# In[ ]:


#df4.filter(col("CharacteristicName") == "Temperature, water").groupBy("ResultMeasure/MeasureUnitCode").count().show()


# In[ ]:


#df4.filter(col("CharacteristicName") == "Oxygen").groupBy("ResultMeasure/MeasureUnitCode").count().show()
#df4.filter(col("CharacteristicName") == "Oxygen").filter(col("ResultMeasure/MeasureUnitCode") == "mg/l").select("ResultMeasureValue").summary().show()
#df4.filter(col("CharacteristicName") == "Oxygen").filter(col("ResultMeasure/MeasureUnitCode") == "% saturatn").select("ResultMeasureValue").summary().show()


# In[ ]:


#df4.filter(col("CharacteristicName") == "Specific conductance").groupBy("ResultMeasure/MeasureUnitCode").count().show()


# In[ ]:


#df4.filter(col("CharacteristicName") == "pH").groupBy("ResultMeasure/MeasureUnitCode").count().show()
#df4.filter(col("CharacteristicName") == "pH").filter(col("ResultMeasure/MeasureUnitCode") == "None").select("ResultMeasureValue").summary().show()
#df4.filter(col("CharacteristicName") == "pH").filter(col("ResultMeasure/MeasureUnitCode") == "std units").select("ResultMeasureValue").summary().show()


# In[ ]:


#temp = df4.filter(col("CharacteristicName") == "pH")
#temp = df4.filter(col("ResultMeasure/MeasureUnitCode") == "std units")


# In[ ]:


#temp = temp.withColumn("pH", temp["ResultMeasureValue"].cast(FloatType())).select("pH")


# In[ ]:


#temp.summary().show()


# In[ ]:


df4 = df4.filter((col("HydrologicEvent") == "Storm") | (col("HydrologicEvent") == "Routine sample"))


# In[ ]:


df4 = df4.filter((col("ResultMeasureUnits") == "std units") |                (col("ResultMeasureUnits") == "deg C") |                (col("ResultMeasureUnits") == "mg/l") |                (col("ResultMeasureUnits") == "uS/cm @25C"))


# In[ ]:


df_characteristic = df4.groupBy("ActivityStartDate", "MonitoringLocationIdentifier").agg(collect_list("CharacteristicName"))


# In[ ]:


df_value = df4.groupBy("ActivityStartDate", "MonitoringLocationIdentifier").agg(collect_list("ResultMeasureValue"))


# In[ ]:


df_hydro_event = df4.groupBy("ActivityStartDate", "MonitoringLocationIdentifier").agg(collect_list("HydrologicEvent"))


# In[ ]:


df5 = df_characteristic.join(df_value, ["ActivityStartDate", "MonitoringLocationIdentifier"]).join(df_hydro_event, ["ActivityStartDate", "MonitoringLocationIdentifier"])


# In[ ]:


#df5.show()


# In[ ]:


features = ["Temperature, water", "Specific conductance", "pH", "Oxygen", "HydrologicEvent", "ResultMeasure/MeasureUnitCode"]


# In[ ]:


df6 = df5.withColumn(features[0],                     when(array_position("collect_list(CharacteristicName)", features[0]) != 0,                         col("collect_list(ResultMeasureValue)")[array_position("collect_list(CharacteristicName)", features[0]) - 1]
                         ))


# In[ ]:


df7 = df6.withColumn(features[1],                     when(array_position("collect_list(CharacteristicName)", features[1]) != 0,                         col("collect_list(ResultMeasureValue)")[array_position("collect_list(CharacteristicName)", features[1]) - 1]
                         ))


# In[ ]:


df8 = df7.withColumn(features[2],                     when(array_position("collect_list(CharacteristicName)", features[2]) != 0,                         col("collect_list(ResultMeasureValue)")[array_position("collect_list(CharacteristicName)", features[2]) - 1]
                         ))


# In[ ]:


df9 = df8.withColumn(features[3],                     when(array_position("collect_list(CharacteristicName)", features[3]) != 0,                         col("collect_list(ResultMeasureValue)")[array_position("collect_list(CharacteristicName)", features[3]) - 1]
                         ))


# In[ ]:


df10 = df9.withColumn(features[4],                     when(array_position("collect_list(HydrologicEvent)", "Storm") != 0, 1).otherwise(0))


# In[ ]:


#df10.show(5)


# In[ ]:


#df10.columns


# In[ ]:


#df10.filter(col("pH") > 14).select("pH").show()


# In[ ]:


df11 = df10.withColumn("pH", when(df10["pH"] == "None", 0)
                                          .when(df10["pH"] == "Moderate", 5)
                                          .when(df10["pH"] == "Mild", 3)
                                          .otherwise(df10["pH"].cast("float")))


# In[ ]:


# # Convert from string to float
df12 = df11.withColumn("Temperature, water", df10["Temperature, water"].cast(FloatType()))
df13 = df12.withColumn("Specific conductance", df11["Specific conductance"].cast(FloatType()))
df14 = df13.withColumn("pH", df12["pH"].cast(FloatType()))
df15 = df14.withColumn("Oxygen", df13["Oxygen"].cast(FloatType()))
df16 = df15.withColumn("HydrologicEvent", df14["HydrologicEvent"].cast(FloatType()))


# In[ ]:


# Define the pH scale range
min_ph = 0
max_ph = 14

# Filter out rows with pH values outside the pH scale range
df16 = df16.filter((col("pH") >= min_ph) & (col("pH") <= max_ph))

# Filter out rows with extreme water temperatures, i.e > 80 C
df16 = df16.filter(col("Temperature, water").cast("float") <= 80)

df16 = df16.filter(col("Specific conductance").cast("float") <= 100000)

df16 = df16.filter(col("Oxygen").cast("float") <= 100)


# In[ ]:


#df16.show()


# In[ ]:


df17 = df16.filter(col("pH").isNotNull())


# In[ ]:


df18 = df17.select("Temperature, water", "Specific conductance", "Oxygen", "HydrologicEvent", "pH")


# In[ ]:


#df18.printSchema()


# In[ ]:


#df18.show()


# In[ ]:


#df18.count()


# # Modeling

# In[8]:


ml_df = pd.DataFrame(columns = ["Model", "RMSE", "R2", "Method"])


# ## Method 2: Replace NULL values with mean in corresponding col AND county

# In[75]:


MAP_MAKER_API_KEY = "6618272817df1758841685bay0ce314"

# Latitude, longitude retrieval via NLDI API with USGS Code
## Note: No request limitations
def retrieve_coordinates(usgs_code):
    response = requests.get('https://labs.waterdata.usgs.gov/api/nldi/linked-data/wqp/' + usgs_code)
    
    if response.status_code == 200:
        data = response.json()  # Convert response to JSON format
        if data:
            # Accessing coordinates
            coordinates = data['features'][0]['geometry']['coordinates']

            # Storing latitude and longitude
            latitude = str(coordinates[1])
            longitude = str(coordinates[0])
            return coordinates
    else:
        return "NULL"
        
# Reverse geocoding via map maker API
## Note: 1 mill request/mo for free
def retrieve_state(coordinates):
    if coordinates != "NULL":
        latitude = str(coordinates[1])
        longitude = str(coordinates[0])

        response = requests.get("https://geocode.maps.co/reverse?lat=" + latitude + "&lon=" + longitude + "&api_key=" + MAP_MAKER_API_KEY)
        if response.status_code == 200:
            data = response.json()
            if 'state' in data['address']:
                state = data['address']['state']
                return state
    return "NULL"


# In[76]:


# Adding state column to each individual sample in dataset
states_column = []
pandasDf = df17.toPandas()

for index, row in pandasDf.iterrows():
    # Access each row's data
    monitoring_location_id = row['MonitoringLocationIdentifier']
    coords = retrieve_coordinates(monitoring_location_id)
    if coords == "NULL":
        states_column.append(coords)
    else:
        state = retrieve_state(coords)
        states_column.append(state)


# In[77]:


# Train/test split
pandasDf["State"] = states_column
spark_pandas_df = ss.createDataFrame(pandasDf)

(training_data, validation_data, test_data) = spark_pandas_df.randomSplit([0.6, 0.2, 0.2], seed = 123)


# In[78]:


pandas_training = training_data.toPandas()
pandas_validation = validation_data.toPandas()
pandas_test = test_data.toPandas()

# Calculate mean values by county
mean_values_by_state = pandas_training.groupby("State").agg(
    {"Oxygen": "mean", "Temperature, water": "mean", "Specific conductance": "mean"}
).reset_index()

# Print the mean values by county
#print(mean_values_by_state)
  
for index, row in pandas_training.iterrows():
    state = row["State"]
    for characteristic in ["Oxygen", "Temperature, water", "Specific conductance"]:
        if row[characteristic] == 0:
            if state in mean_values_by_state["State"].values:
                mean_value = mean_values_by_state.loc[mean_values_by_state["State"] == state, characteristic].values[0]
            else:
                # If the state is not found, use the mean values for "Null"
                mean_value = mean_values_by_state.loc[mean_values_by_state["State"] == "NULL", characteristic].values[0]
            pandas_training.at[index, characteristic] = mean_value
    
for index, row in pandas_validation.iterrows():
    state = row["State"]
    for characteristic in ["Oxygen", "Temperature, water", "Specific conductance"]:
        if row[characteristic] == 0:
            if state in mean_values_by_state["State"].values:
                mean_value = mean_values_by_state.loc[mean_values_by_state["State"] == state, characteristic].values[0]
            else:
                # If the state is not found, use the mean values for "Null"
                mean_value = mean_values_by_state.loc[mean_values_by_state["State"] == "NULL", characteristic].values[0]
            pandas_validation.at[index, characteristic] = mean_value
        
for index, row in pandas_test.iterrows():
    state = row["State"]
    for characteristic in ["Oxygen", "Temperature, water", "Specific conductance"]:
        if row[characteristic] == 0:
            if state in mean_values_by_state["State"].values:
                mean_value = mean_values_by_state.loc[mean_values_by_state["State"] == state, characteristic].values[0]
            else:
                # If the state is not found, use the mean values for "Null"
                mean_value = mean_values_by_state.loc[mean_values_by_state["State"] == "NULL", characteristic].values[0]
            pandas_test.at[index, characteristic] = mean_value
        
spark_training_df = ss.createDataFrame(pandas_training)
spark_validation_df = ss.createDataFrame(pandas_validation)
spark_test_df = ss.createDataFrame(pandas_test)

spark_training_data = spark_training_df.select("Temperature, water", "Specific conductance", "Oxygen", "HydrologicEvent", "pH")
spark_validation_data = spark_validation_df.select("Temperature, water", "Specific conductance", "Oxygen", "HydrologicEvent", "pH")
spark_test_data = spark_test_df.select("Temperature, water", "Specific conductance", "Oxygen", "HydrologicEvent", "pH")


# ### Standardization

# In[85]:


training_input_columns = spark_training_data.columns[:-1]
training_output_columns = spark_training_data.columns[-1]
validation_input_columns = spark_validation_data.columns[:-1]
validation_output_columns = spark_validation_data.columns[-1]
test_input_columns = spark_test_data.columns[:-1]
test_output_columns = spark_test_data.columns[-1]


# In[ ]:


# Create a feature vector by combining all feature columns into a single 'features' column
training_assembler = VectorAssembler(inputCols = training_input_columns, outputCol = 'features')
validation_assembler = VectorAssembler(inputCols = validation_input_columns, outputCol = 'features')
test_assembler = VectorAssembler(inputCols = test_input_columns, outputCol = 'features')

training_data = training_assembler.transform(spark_training_data)
validation_data = validation_assembler.transform(spark_validation_data)
test_data = test_assembler.transform(spark_test_data)


# In[ ]:


# Scale the feature vector using StandardScaler
scaler = StandardScaler(inputCol = "features", outputCol = "scaled_features", withStd = True, withMean = True)

training_scaler_model = scaler.fit(training_data)
validation_scaler_model = scaler.fit(validation_data)
test_scaler_model = scaler.fit(test_data)

training_data = training_scaler_model.transform(training_data)
validation_data = validation_scaler_model.transform(validation_data)
test_data = test_scaler_model.transform(test_data)


# ### Ridge Linear Regression

# In[80]:


# Training
model = LinearRegression(featuresCol = "scaled_features", labelCol = "pH", predictionCol = "predicted_pH", elasticNetParam = 0)
lr_model = model.fit(training_data)


# In[86]:


lr_predictions = lr_model.transform(validation_data)

lr_evaluator_rmse = RegressionEvaluator(predictionCol = "predicted_pH",                  labelCol = "pH", metricName = "rmse")
lr_evaluator = RegressionEvaluator(predictionCol = "predicted_pH",                  labelCol = "pH", metricName = "r2")



ml_df.loc[len(ml_df)] = {"Model": "Ridge Regression", "RMSE": lr_evaluator_rmse.evaluate(lr_predictions), "R2": lr_evaluator.evaluate(lr_predictions), "Method": 2}

#print("RMSE on validation data : %g" % lr_evaluator_rmse.evaluate(lr_predictions))
#print("r2 on validation data : %g" % lr_evaluator.evaluate(lr_predictions))


# In[ ]:


coefficients = lr_model.coefficients.toArray().tolist()
intercept = [lr_model.intercept]
data = {"Temperature, water": coefficients[0], "Specific conductance": coefficients[1], "Oxygen": coefficients[2], "HydrologicEvent": coefficients[3], "Intercept": intercept}

coefficients_df = pd.DataFrame(data)

#print(coefficients_df)


# In[ ]:


output_path = "/storage/home/trn5106/work/Final Project/main code/Linear_Coefficients.csv"
coefficients_df.to_csv(output_path)


# ### Decision Tree

# In[82]:


# Testing
dt = DecisionTreeRegressor(featuresCol = "scaled_features", labelCol = "pH", predictionCol = "predicted_pH")
dt_model = dt.fit(training_data)
dt_predictions = dt_model.transform(test_data)

dt_evaluator_rmse = RegressionEvaluator(predictionCol="predicted_pH",                  labelCol="pH",metricName="rmse")
dt_evaluator = RegressionEvaluator(predictionCol="predicted_pH",                  labelCol="pH",metricName="r2")

ml_df.loc[len(ml_df)] = {"Model": "Decision Tree", "RMSE": dt_evaluator_rmse.evaluate(dt_predictions), "R2": dt_evaluator.evaluate(dt_predictions), "Method": 2}


#print("RMSE on test data : %g" % dt_evaluator_rmse.evaluate(dt_predictions))
#print("r2 on test data : %g" % dt_evaluator.evaluate(dt_predictions))


# ### Random Forest

# In[58]:


from pyspark.ml.regression import RandomForestRegressor
rf = RandomForestRegressor(featuresCol = 'scaled_features', labelCol = 'pH', predictionCol = "predicted_pH")
rf_model = rf.fit(training_data)
rf_predictions = rf_model.transform(test_data)

rf_evaluator_rmse = RegressionEvaluator(predictionCol = "predicted_pH",                  labelCol = "pH", metricName = "rmse")
rf_evaluator = RegressionEvaluator(predictionCol = "predicted_pH",                  labelCol = "pH", metricName = "r2")

ml_df.loc[len(ml_df)] = {"Model": "Random Forest", "RMSE": rf_evaluator_rmse.evaluate(rf_predictions), "R2": rf_evaluator.evaluate(rf_predictions), "Method": 2}

#print("RMSE on validation data : %g" % rf_evaluator_rmse.evaluate(rf_predictions))
#print("r2 on validation data : %g" % rf_evaluator.evaluate(rf_predictions))


# ### Gradient Boosted Tree Regression

# In[84]:


from pyspark.ml.regression import GBTRegressor
# Define GBTRegressor model
gbt = GBTRegressor(featuresCol = 'scaled_features', labelCol = 'pH', predictionCol = "predicted_pH")

# Fit the model
gbt_model = gbt.fit(training_data)

gbt_predictions = gbt_model.transform(test_data)

gbt_evaluator_rmse = RegressionEvaluator(predictionCol = "predicted_pH",                  labelCol = "pH", metricName = "rmse")
gbt_evaluator = RegressionEvaluator(predictionCol = "predicted_pH",                  labelCol = "pH", metricName = "r2")

ml_df.loc[len(ml_df)] = {"Model": "Gradient Boosted Tree Regression", "RMSE": gbt_evaluator_rmse.evaluate(gbt_predictions), "R2": gbt_evaluator.evaluate(gbt_predictions), "Method": 2}

#print("RMSE on validation data : %g" % gbt_evaluator_rmse.evaluate(gbt_predictions))
#print("r2 on validation data : %g" % gbt_evaluator.evaluate(gbt_predictions))


# In[ ]:


output_path = "/storage/home/trn5106/work/Final Project/main code/ML_Models.csv"
ml_df.to_csv(output_path)


# ### Hyperparameter Tuning

# In[ ]:


from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
dt = DecisionTreeRegressor(featuresCol = "scaled_features", labelCol = "pH", predictionCol = "predicted_pH")

paramGrid = (ParamGridBuilder()
             .addGrid(dt.maxDepth, [1, 2, 3, 4, 5, 6])  # max depth of the tree
             .addGrid(dt.minInstancesPerNode, [1, 2, 3, 4, 5, 6, 7, 8])  # minimum number of instances each child must have
             .build())

evaluator = RegressionEvaluator(predictionCol = "predicted_pH", labelCol = "pH", metricName = "rmse")
evaluator_r2 = RegressionEvaluator(predictionCol = "predicted_pH", labelCol = "pH", metricName = "r2")

crossval = CrossValidator(estimator = dt,
                          estimatorParamMaps = paramGrid,
                          evaluator = evaluator,
                          numFolds = 3)

cvModel = crossval.fit(training_data)

best_model = cvModel.bestModel

best_model.save('./Best_DT')

predictions = best_model.transform(validation_data)
rmse = evaluator.evaluate(predictions)
r2 = evaluator_r2.evaluate(predictions)

#print("RMSE on validation data : %g" % rmse)
#print("r2 on validation data : %g" % r2)


# In[ ]:


params = [{p.name: v for p, v in m.items()} for m in cvModel.getEstimatorParamMaps()]

dt_hpt = pd.DataFrame.from_dict([
    {cvModel.getEvaluator().getMetricName(): metric, **ps} 
    for ps, metric in zip(params, cvModel.avgMetrics)
])


# In[ ]:


output_path = "/storage/home/trn5106/work/Final Project/main code/DT_HPT_Local.csv"
dt_hpt.to_csv(output_path)


# ### Testing

# In[ ]:


final_df = pd.DataFrame(columns = ["Model", "RMSE", "R2", "Method"])


# In[ ]:


# Training
dt = DecisionTreeRegressor(featuresCol = "scaled_features", labelCol = "pH",                            predictionCol = "predicted_pH", maxDepth = best_model.getMaxDepth(),                            minInstancesPerNode = best_model.getMinInstancesPerNode())
dt_model = dt.fit(training_data)


# In[ ]:


# Testing
dt_predictions = dt_model.transform(test_data)

dt_evaluator_rmse = RegressionEvaluator(predictionCol = "predicted_pH",                  labelCol = "pH", metricName = "rmse")
dt_evaluator = RegressionEvaluator(predictionCol = "predicted_pH",                  labelCol = "pH", metricName = "r2")

final_df.loc[len(final_df)] = {"Model": "Decision Tree", "RMSE": dt_evaluator_rmse.evaluate(dt_predictions), "R2": dt_evaluator.evaluate(dt_predictions), "Method": 2}

#print("RMSE on test data : %g" % dt_evaluator_rmse.evaluate(dt_predictions))
#print("r2 on test data : %g" % dt_evaluator.evaluate(dt_predictions))


# In[ ]:


output_path = "/storage/home/trn5106/work/Final Project/main code/Final_Model.csv"
final_df.to_csv(output_path)

