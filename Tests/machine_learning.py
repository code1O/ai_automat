#         Essential for tests compiling
# ===================================================
import sys
import os

dir_ = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.abspath(dir_))
# ===================================================

import numpy as np
import matplotlib.pyplot as plt
import Make_AI as machle

data_file = "Data/machle_demo_data.csv"
categories = ["Volume", "Weight"]
predict_categorie = "CO2"
values = [240, 750]

predict_instance = machle.prediction(data_file, categories, predict_categorie)
transformed_predict = predict_instance.execute_transform(values)
print(transformed_predict)