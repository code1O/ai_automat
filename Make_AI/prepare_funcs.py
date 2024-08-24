#         Essential for compiling
# ===================================================
import sys
import os

path_folder = "../"

dir_ = os.path.join(os.path.dirname(__file__), path_folder)
sys.path.insert(0, os.path.abspath(dir_))
# ===================================================

from python_utils import (
    _Typedata, _TypeNum,
)

import torchtext
torchtext.disable_torchtext_deprecation_warning()

import numpy as np
import pandas as pd
import tensorflow as tf
import torch.nn as nn
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler

scale = StandardScaler()
poly_regress = lambda x, y, n: np.poly1d(np.polyfit(x,y,n))

class neural_networks:
    """
    # Neural Networks
    Init a neural network with both `tensorflow` or `pytorch`
    
    ## [TensorFlow](https://github.com/tensorflow/tensorflow)
    
    ```python
    import numpy as np
    
    input_data = np.array([100.0])
    input_shape = [1,]
    input_units = 2
    
    celsius = np.array([40, 10, -15, 11], dtype=float)
    farenheit = np.array([104, 50, 5, 51], dtype=float)
    
    neural_net = neural_networks(input_data, celsius, farenheit, input_units, input_shape)
    tf_neural_net = neural_net.tensorflow(rounds=400)
    
    predicted_result = tf_neural_net["result"]
    predicted_loss = tf_neural_net["loss"]
    
    print(predicted_result, predicted_loss, sep="\\n")
    
    ```
    
    ## [PyTorch](https://github.com/pytorch/pytorch)
    
    ```python
    
    import torch.nn as nn
    import numpy as np
    
    array_a = np.array([1,2,3,4,5,6,7,8,9.10], dtype=float, ndim=2)
    array_b = np.array([[11,12,13,14,15,16,17,18,19,20]], dtype=float, ndmin=2)
    
    tensor_a = torch.from_numpy(array_a)
    tensor_b = torch.from_numpy(array_b)
    
    input_tensor = tensor_a * tensor_b
    
    LinearSeq = nn.Sequential(
    nn.Linear(10, 18),
    nn.Linear(10, 18),
    nn.Linear(20, 5)
    )
    
    neural_net = neural_networks
    
    out_features = 1
    
    torch_neural_net = neural_net.pytorch(input_tensor, out_features, LinearSeq)
    first_model_values = torch_neural_net[0]
    print(first_model_values["model"])
    
    ```
    
    """
    def __init__(self, array_a, array_b) -> None:
        self.array_a, self.array_b = array_a, array_b
    
    def tensor_flow(self, rounds, tf_optimizer=tf.keras.optimizers.Adam, optimizer_value=0.1, **kwargs):
        """
        **Parameters**
        
        - `tf_optimizer`
          
          Set `tf.keras.optimizers.Adam` as default.
          
          you can replace it with other optimizer
        
        - `optimizer_value`
          
          Set 0.1 as default value to optimize the compilation.
          
          you can replace it as many as you want
        
        **Kwargs**
        - `input_shape`
          
          The shape of the array, you can know this by ``numpy.shape``
        
        - `input_units`
        
          The quantity of elements inside an array
          
          `units = len(input_array)`
          
          Consider that both `x` & `y` arrays must have the same quantity of elements
          
        """
        
        kwargs_params = {
            "input_data": _Typedata,
            "input_shape": _Typedata,
            "input_units": _TypeNum
        }
        
        kwargs_params.update(kwargs)
        
        input_data = kwargs_params["input_data"]
        input_shape, input_units = kwargs_params["input_shape"], kwargs_params["input_units"]
        
        layer_hide1 = tf.keras.layers.Dense(units=input_units, input_shape=input_shape)
        layer_hide2 = tf.keras.layers.Dense(units=input_units)
        output_layer = tf.keras.layers.Dense(units=1)
        model = tf.keras.Sequential([
            layer_hide1, layer_hide2, output_layer,
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Flatten()
        ])
        model.compile(
            optimizer=tf_optimizer(optimizer_value),
            loss="mean_squared_error"
        )
        history_ = model.fit(self.array_a, self.array_b, epochs=rounds, verbose=False)
        result = model.predict(input_data)
        loss = history_.history["loss"]
        return dict(history_loss=loss, result=result)
    
    def pytorch(input_tensor, out_features, sequential):
        linear_layer = nn.Linear(in_features=len(input_tensor), out_features=out_features)
        sequential_layer = sequential
        model_1, model_2 = linear_layer(input_tensor), sequential_layer(input_tensor)
        device, dtype = linear_layer.device, linear_layer.dtype
        dictionary_model_1 = dict(device=device, dtype=dtype, model_1=model_1)
        return dictionary_model_1, model_2

class HardPredict:
    def __init__(self, units, x, y, channels: int = 1) -> None: 
        
        self.channels = channels
        self.x, self. y = x, y
        self.input_units = units
    
    @tf.function
    def prepared_function(self, dropout_value: float=0.5, tf_optimizer: str = "adam", 
                   cores: int = 32, *,
                   group_shape: _Typedata, matrix_shape: _Typedata, **kwargs):
        r"""
        :Keyword Arguments:
            * *epochs* (``int``) --
            
              Recommended at least 60 epochs for better optimization
        """
        hide_params = {
            "input_data": _Typedata,
            "size_batch": int,
            "data_validation": _Typedata,
            "epochs": int,
            "steps_per_epoch": _TypeNum,
            "validation_steps": _TypeNum
        }
        hide_params.update(kwargs)
        model_epochs, data_steps_epoch = hide_params["epochs"], hide_params["steps_per_epoch"]
        input_data = hide_params["input_data"]
        size_batch = hide_params["size_batch"]
        data_validation, steps_validation = hide_params["data_validation"], hide_params["validation_steps"]
        aumented_core = cores * 2
        x, y, channels, input_units = (self.x, self.y, self.channels, self.input_units)
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(cores, group_shape, input_shape=(x, y, channels), activation="relu"),
            tf.keras.layers.MaxPooling2D(matrix_shape),
            tf.keras.layers.Conv2D(aumented_core, group_shape, activation="relu"),
            tf.keras.layers.MaxPooling2D(matrix_shape),
            tf.keras.layers.Dropout(dropout_value),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=input_units, activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax")
        ])
        model.compile(
            optimizer=tf_optimizer,
            loss=tf.keras.losses.SparseCategoricalCrossentropy(),
            metrics=['accuracy']
        )
        
        autotune = tf.data.experimental.AUTOTUNE
        input_data = input_data.cache().shuffle(1000).batch(size_batch).prefetch(buffer_size=autotune)
        data_validation = data_validation.cache().batch(size_batch).prefetch(buffer_size=autotune)
        
        history = model.fit(
            input_data,
            epochs=model_epochs,
            batch_size=size_batch,
            validation_data=data_validation,
            steps_per_epoch=data_steps_epoch,
            validation_steps=steps_validation,
            verbose=False
        )
        
        return history
        

class Prediction:
    """
    Sklearn prediction
    
    Using sklearn to predict values by neural networks
    
    """
    def __init__(self, input_data, input_categorie, output_categorie):
        self.categories = dict(
            input=input_categorie,
            output=output_categorie
        )
        self.input_data = input_data
    
    @property
    def execute_normal(self):
        X, y = self.categories
        regression = linear_model.LinearRegression()
        fit_regression = regression.fit(X, y)
        coeficient = fit_regression.coef_
        predict = fit_regression.predict([self.input_data])
        dictionary_results = dict(coef=coeficient, predict=predict)
        return dictionary_results
    
    @property
    def execute_transforming(self):
        X, y = self.categories
        scaledX = scale.fit_transform(X)
        regression = linear_model.LinearRegression()
        fit_regression = regression.fit(scaledX, y)
        scaled_input_data = scale.transform(self.input_data)
        scaled_array = scaled_input_data[0]
        result = fit_regression.predict([scaled_array])
        dictionary = dict(to_predict=self.input_data, value_predicted=result)
        return dictionary
    