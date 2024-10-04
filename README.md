
### Developed by: SHARAN MJ
### Register no:212222240097
### Date: 
# Ex.No: 6               HOLT WINTERS METHOD




### AIM: 
To implement Holt-Winters model on amazon stock Price Data Set and make future predictions

### ALGORITHM:
1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as
datetime, and perform some initial data exploration
3. You group the data by date and resample it to a monthly frequency (beginning of the month
4. You plot the time series data
5. You import the necessary 'statsmodels' libraries for time series analysis
6. You decompose the time series data into its additive components and plot them:
7. You calculate the root mean squared error (RMSE) to evaluate the model's performance
8. You calculate the mean and standard deviation of the entire sales dataset, then fit a Holt-
Winters model to the entire dataset and make future predictions
9. You plot the original sales data and the predictions
### PROGRAM:
```py
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd

from google.colab import drive
drive.mount('/content/drive')

!ls "/content/drive/MyDrive/time series" # List files in the directory to verify the file exists and its name

data = pd.read_csv('/content/drive/MyDrive/time series/AMZN.csv', index_col='Date', parse_dates=True)

# It seems like you might be working with stock market data
# and want to resample the 'Low' column. 
# Try replacing 'Min' with 'Low' if this is the case.

data = data.resample('MS').mean()

# Scaling the Data using MinMaxScaler 
scaler = MinMaxScaler()
data_scaled = pd.Series(scaler.fit_transform(data.values.reshape(-1, 1)).flatten(), index=data.index)

# Split into training and testing sets (80% train, 20% test)
train_data = data_scaled[:int(len(data_scaled) * 0.8)]
test_data = data_scaled[int(len(data_scaled) * 0.8):]

fitted_model_add = ExponentialSmoothing(
    train_data, trend='add', seasonal='add', seasonal_periods=12
).fit()

# Forecast and evaluate
test_predictions_add = fitted_model_add.forecast(len(test_data))

# Evaluate performance
print("MAE :", mean_absolute_error(test_data, test_predictions_add))
print("RMSE :", mean_squared_error(test_data, test_predictions_add, squared=False))

# Plot predictions
plt.figure(figsize=(12, 8))
plt.plot(train_data, label='TRAIN', color='black')
plt.plot(test_data, label='TEST', color='green')
plt.plot(test_predictions_add, label='PREDICTION', color='red')
plt.title('Train, Test, and Additive Holt-Winters Predictions')
plt.legend(loc='best')
plt.show()

final_model = ExponentialSmoothing(data, trend='mul', seasonal='mul', seasonal_periods=12).fit()

# Forecast future values
forecast_predictions = final_model.forecast(steps=12)

data.plot(figsize=(12, 8), legend=True, label='Current Price')
forecast_predictions.plot(legend=True, label='Forecasted Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title(' Forecast')
plt.show()


```

### OUTPUT:

EVALUATION

![Screenshot 2024-10-04 092933](https://github.com/user-attachments/assets/fe24374d-59f7-4296-a936-a685515b1671)


TEST_PREDICTION

![download (1)](https://github.com/user-attachments/assets/b466dadd-45db-45fd-bd34-248bad3c356b)




FINAL_PREDICTION
![download](https://github.com/user-attachments/assets/4a41377d-0b36-4f9c-83e8-040ae2b8cd90)


### RESULT:
Therefore a python program has been executed successfully based on the Holt Winters Method model.
