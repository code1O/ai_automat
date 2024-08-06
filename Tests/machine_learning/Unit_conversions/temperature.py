#         Essential for tests compiling
# ===================================================
import sys
import os

dir_ = os.path.join(os.path.dirname(__file__), "../../../")
sys.path.insert(0, os.path.abspath(dir_))
# ===================================================

clear = lambda: os.system("cls")

import numpy as np
import pandas
import torch
from Make_AI import neural_networks, mathematics

df = pandas.read_json("data/machle_data.json")
unit_conv = df.get("Unit_conversions")
temperature = unit_conv[0]["Temperature"]
celsius = np.array(temperature[0]["Celsius"], dtype=float)
kelvin = np.array(temperature[0]["Kelvin"], dtype=float)

input_data = np.array([215.0])
input_shape = (1,)
input_units = 16

neural_net_model = neural_networks(celsius, kelvin)
tf_neural_net = neural_net_model.tensor_flow(
    rounds=400, optimizer_value=1.0,
    input_data=input_data, input_shape=input_shape,
    input_units=input_units
)
tf_result = tf_neural_net["result"]
adjust_array_result = np.array(tf_result[0])

num_to_calc = np.array([2 * 10 ** (-5)]) 

tensor_a = torch.from_numpy(adjust_array_result)
tensor_b = torch.from_numpy(num_to_calc)
tensor_c = tensor_a - tensor_b

predicted_value = tensor_c
expected_value = 488.1500

math_instance = mathematics(celsius, kelvin)
precission = math_instance.precission(
    predicted_value, expected_value, 
    input_units, input_shape
)

clear()
print(tensor_c, precission, sep="\n")