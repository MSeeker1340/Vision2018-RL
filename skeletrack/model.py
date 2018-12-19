import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import mean_squared_error

# Define a 5-layer fully connected NN to learn the policy
policy_model = Sequential([
    Dense(units=78, activation='relu', input_dim=78),
    Dense(units=60, activation='relu'),
    Dense(units=60, activation='relu'),
    Dense(units=39, activation='relu'),
    Dense(units=39)
])

# Define the custom loss function
def weighted_mse(y_true, y_pred):
    mse = mean_squared_error(y_true[...,:-1], y_pred)
    weight = y_true[...,-1]
    return weight * mse

# Compile the model
policy_model.compile(loss=weighted_mse, optimizer='sgd')
