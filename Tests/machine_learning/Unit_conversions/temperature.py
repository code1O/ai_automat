#         Essential for tests compiling
# ===================================================
import sys
import os

path_folder = "../../../"

dir_ = os.path.join(os.path.dirname(__file__), path_folder)
sys.path.insert(0, os.path.abspath(dir_))
# ===================================================

clear = lambda: os.system("cls")

import numpy as np
import pandas
import torch
from Make_AI import (model_conversion, globalReadJson)

input_value = [215.0]
num_to_calc = np.array([9 * 10])

conversion = model_conversion("Models/prediction", "model.h5", "Temperature", "Celsius", "Farenheit", input_value)

print(conversion, sep="\n")

