##############################################
##### Predicting EUR/USD pair using LSTM #####
##############################################

###################################
### Part 1 - Data Preprocessing ###
###################################

### Importing the libraries ###
import numpy as np
import pandas as pd

### Importing the data set ###
dataset = pd.read_csv('dataset.csv')

### Set basic parameters ###
timesteps = 120
test_size = 0.2     # 0.2 = 20% of the dataset

### Set hyperparameters ###
from keras.optimizers import Adam
parameters = {'hidden_layers': [3, 6],
              'units_per_layer': [50, 100, 200],
              'dropout': [0.0, 0.2, 0.4],
              'batch_size': [128, 256],
              'epochs': [100],
              'optimizer': [Adam(lr = 0.001)],
              'loss': ['mean_squared_error'],
              'metrics': ['accuracy']}  

### Processing the specific dataset ###
# The code an next assumes that the prediction(y) is the last column of the dataset.
# If your dataset isn't ready, process it here.

# Convert dates to days
# 0 is Monday - 4 is Friday
# Stock exchanges are closed on Weekends
import datetime
for i in range (0, dataset.shape[0]):
    dataset.iloc[i,4] = datetime.datetime.strptime(dataset.iloc[i,4], '%m/%d/%Y').weekday()
    
# We don't need the 2 last columns and we have to make 'Price' column being the last column.
# Swap 'Price' and "RSI' columns
for i in range (0, dataset.shape[0]):
    dataset.iloc[i,16] = dataset.iloc[i,3]
    dataset.iloc[i,3] = dataset.iloc[i,15]
    dataset.iloc[i,15] = dataset.iloc[i,16]
# Delete the unused columns
dataset = dataset.iloc[:,:16]

### Feature Scaling - Normalization ###
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
dataset_scaled = sc.fit_transform(dataset)

### Creating a 3D data structure with [timesteps] timesteps and one output ###
# [Samples, Timesteps, Features]
# x_train(Z) = [Features(Z-1)]
# y_train(Z) = [Feature(Z)]
x = []
y = []
for i in range(timesteps, dataset_scaled.shape[0]):
    x.append(dataset_scaled[i-timesteps:i, :dataset_scaled.shape[1]-1])
    y.append(dataset_scaled[i, dataset_scaled.shape[1]-1])
x, y = np.array(x), np.array(y)
y = np.reshape(y, (y.shape[0], 1))

### Splitting the dataset into the Training set and Test set ###
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = test_size, random_state = 0)


##################################
### Part 2 - Building the LSTM ###
##################################

### Importing the Keras libraries and packages ###
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

### Build the regressor ###         
def build_regressor(hidden_layers, units_per_layer, dropout, optimizer, loss, metrics):
    # Initialising the LSTM
    regressor = Sequential()    
    # Adding the first LSTM layer and some Dropout regularisation
    regressor.add(LSTM(units = units_per_layer, return_sequences = True, input_shape = (x_train.shape[1], x_train.shape[2])))
    regressor.add(Dropout(dropout))    
    # Adding new LSTM hidden layers if needed
    for i in range(0, hidden_layers-1):
        regressor.add(LSTM(units = units_per_layer, return_sequences = True))
        regressor.add(Dropout(dropout))
    # Adding the pre-last LSTM layer
    regressor.add(LSTM(units = units_per_layer))
    regressor.add(Dropout(dropout))    
    # Adding the output layer
    regressor.add(Dense(units = 1))    
    # Compiling the LSTM
    regressor.compile(optimizer = optimizer, loss = loss, metrics = metrics)    
    return regressor

### Train the model ###         
def fit_regressor(epochs, batch_size):
    return regressor.fit(x_train, y_train, epochs = epochs, batch_size = batch_size, validation_data=(x_test, y_test),  shuffle=True)

### Start Evaluating and Tuning our LSTM model ###
import matplotlib.pyplot as plt
results = []
best_parameters = []
best_loss = float("inf")
best_model = Sequential()  
for layers in parameters["hidden_layers"]:
    for units_per_layer in parameters["units_per_layer"]:
        for dropout in parameters["dropout"]:
            for batch_size in parameters["batch_size"]:
                for epochs in parameters["epochs"]:
                    for optimizer in parameters["optimizer"]:
                        for loss in parameters["loss"]:
                            for metrics in parameters["metrics"]:
                                regressor = build_regressor(int(layers), units_per_layer, dropout, optimizer, loss, [metrics])
                                history = fit_regressor(epochs, batch_size)
                                results.append([layers, units_per_layer, dropout, batch_size, epochs, optimizer, loss, metrics, 
                                                float(history.history['loss'][0]), float(history.history['val_loss'][0])])                                
                                plt.plot(history.history['val_loss'][2:epochs], color = 'blue', label = 'Test')
                                plt.plot(history.history['loss'][2:epochs], color = 'red', label = 'Train')
                                plt.xlabel('Epochs')
                                plt.ylabel('Error')
                                plt.legend()
                                plt.show()
                                print('Layers:\t\t',layers,'\nUnits per layer:',units_per_layer,'\nDropout:\t',dropout,'\nBatch size:\t', batch_size, 
                                      '\nEpochs:\t\t',epochs,'\nOptimizer:\t',optimizer,'\nLoss function:\t',loss,'\nMetrics:\t',metrics,
                                      '\nLoss (Train):\t',history.history['loss'][epochs-1],'\nLoss (Test):\t',history.history['val_loss'][epochs-1],'\n\n')
                                # Keep the best model
                                if float(history.history['loss'][epochs-1]) < best_loss:
                                    best_model = regressor                                 
                                    best_loss = float(history.history['loss'][0])
                                    best_parameters.clear()
                                    best_parameters.append([layers, units_per_layer, dropout, batch_size, epochs, optimizer, loss, metrics,
                                                           float(history.history['loss'][0]), float(history.history['val_loss'][0]),
                                                           float(history.history['acc'][0]), float(history.history['val_acc'][0])])
                                
### Show the best parameters ###
print('************* Best parameters *************')
print('* Layers:\t',best_parameters[0][0],'\n* Units:\t',best_parameters[0][1],'\n* Dropout:\t',best_parameters[0][2],'\n* Batch size:\t', 
      best_parameters[0][3],'\n* Epochs:\t',best_parameters[0][4],'\n* Optimizer:\t',best_parameters[0][5],'\n* Loss function:',best_parameters[0][6],
      '\n* Metrics:\t',best_parameters[0][7],'\n* Loss (Train):\t',best_parameters[0][8],'\n* Loss (Test):\t',best_parameters[0][9])

print('\n*******************************************\n')

### Save the weights ###
best_model.save_weights('./checkpoint')

###########################################
### Part 3 - Making a single prediction ###
###########################################

### INSERT HERE your timeseries in this array [Timesteps]x[Features]###
for_predict = x_test[0,:] # For example, take the first timeseries of the Test set

### Reshape and predict ###
# It will use the best trained regressor #
for_predict = np.reshape(for_predict, (1,for_predict.shape[0], for_predict.shape[1]))
predictions_scaled = best_model.predict(for_predict)

### Invert MinMax transform ###
# Our scaler have used a specific array size.
# We have to add some padding to be able to inverse the transform correctly.
padding = np.zeros((for_predict.shape[0],dataset.shape[1]-1))
predictions_scaled = np.append(padding, predictions_scaled, axis=1)
predictions_scaled = sc.inverse_transform(predictions_scaled)
predictions = predictions_scaled[:,dataset_scaled.shape[1]-1]

### Calculate RMSE for the new predictions ###
# ADD HERE the actual values to the actual_values (without normalization)
actual_values = [1.110] # Just an example

# Calculate RMS
from math import sqrt
from sklearn.metrics import mean_squared_error
rmse = sqrt(mean_squared_error(predictions, actual_values))
print('Predictions RMSE: %.3f' % rmse)