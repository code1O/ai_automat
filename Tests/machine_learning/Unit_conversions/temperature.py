#         Essential for tests compiling
# ===================================================
import sys
import os

dir_ = os.path.join(os.path.dirname(__file__), "../../../")
sys.path.insert(0, os.path.abspath(dir_))
# ===================================================

import numpy as np
import pandas
import matplotlib.pyplot as plt
from Make_AI import neural_networks

df = pandas.read_json("data/machle_data.json")
unit_conv = df.get("Unit_conversions")
temperature = unit_conv[0]["Temperature"]
celsius = np.array(temperature[0]["Celsius"], dtype=float)
farenheit = np.array(temperature[0]["Farenheit"], dtype=float)

input_data = np.array([215.0])
input_shape = (1,)
input_units = 16

neural_net = neural_networks(input_data, celsius, farenheit, input_units, input_shape)
tf_neural_net = neural_net.tensor_flow(rounds=300)
tf_result = tf_neural_net["result"]

print(tf_result)