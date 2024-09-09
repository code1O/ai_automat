# Welcome to the machine_learning guidelines

This file contains necessary information about how to work with `Make_AI` functions

## Importing functions in other folders

First, insert this depending the folder where you working

````python
path_folder = "../../"     # Folder_1/Folder_2
path_folder = "../../../"  # Folder_1/Folder_2/Folder_3
````

The string from the variable must be unchanged, in case of changing the string, the interpreter will be confused and not working correctly.

Then, write this code on your location folder. <mark>It could be both with or without comments</mark>

````python

#         Essential for tests compiling
# ===================================================
import sys
import os

dir_ = os.path.join(os.path.dirname(__file__), path_folder)
sys.path.insert(0, os.path.abspath(dir_))
# ===================================================

````

This applies for testing folders and other folders that are not `Make_AI`

## Starting neural network

For starting a neural network you can use `neural_networks` function.

Once you imported the functions, you have to start your neural network

````python

from Make_AI import neural_networks

import numpy as np
import tensorflow as tf
import pandas as pd

df = pd.read_json("data/machle_data.json")
unit_conv = df.get("Unit_conversions")
temperature = unit_conv[0]["Temperature"]
celsius = np.array(temperature[0]["Celsius"])
farenheit = np.array(temperature[0]["Farenheits"])

input_data = np.array([215.0])
input_shape = [1,]
input_units = 16

neural_net = neural_networks(
    input_data, 
    celsius, farenheits,
    input_units, input_shape
)

tf_neural_net = neural_net.tensorflow(rounds=400, optimizer_value=1.0)

predicted_value = tf_neural_net["result"]

print(predicted_value)

````

For more details, check [some testing file](
https://github.com/code1O/ai_automat/blob/main/Tests/machine_learning/Unit_conversions/temperature.py
)