#         Essential for compiling
# ===================================================
import sys
import os

dir_ = os.path.join(os.path.dirname(__file__), "../../../")
sys.path.insert(0, os.path.abspath(dir_))
# ===================================================

import torchtext
torchtext.disable_torchtext_deprecation_warning()

import numpy as np
import pandas
import torch

# `C:/Users/User/.../Make_AI`
from Make_AI import neural_networks, mathematics, Prediction


clear = lambda: os.system("cls")

df = pandas.read_json("data/machle_data.json")
Trigonometric = df.get("Trigonometric")
sines = Trigonometric[0]["sines"]
cosines = Trigonometric[0]["cosines"]
tangents = Trigonometric[0]["tangents"]

warning_message = "Warning: the predicted results could be round"

input_sines, output_sines = (
    np.array(sines[0]["input"], dtype=float),
    np.array(sines[0]["output"], dtype=float)
)

input_cosines, output_cosines = (
    np.array(cosines[0]["input"], dtype=float),
    np.array(cosines[0]["output"], dtype=float)
)

input_tangents, output_tangents = (
    np.array(tangents[0]["input"], dtype=float),
    np.array(tangents[0]["output"], dtype=float)
)

input_shape, input_units = (1,), 16
input_data = np.array([150.35])
expected_sine, expected_cosine, expected_tangent = (
    0.4947, -0.8690, -0.5692
)

num_to_calc = np.array([2 * 10 ** (-5)])

def neural_nets(input_categorie, output_categorie, expected_value):
    neural_net = neural_networks(input_categorie, output_categorie)
    tf_neural_net = neural_net.tensor_flow(
        rounds=400, optimizer_value=0.1,
        input_data=input_data, input_shape=input_shape,
        input_units=input_units
    )
    tf_result = tf_neural_net["result"]
    adjust_array_result = np.array(tf_result[0])
    tensor_a = torch.from_numpy(adjust_array_result)
    tensor_b = torch.from_numpy(num_to_calc)
    tensor_c = tensor_a - tensor_b
    math_instance = mathematics(input_categorie, output_categorie)
    precission = math_instance.precission(
        tensor_c, expected_value, 
        input_units, input_shape
    )
    dictionary = dict(
        predicted_value=tensor_c,
        expected_value=expected_value,
        precission_value = precission
    )
    return dictionary

neural_net_sines = neural_nets(input_sines, output_sines, expected_sine)
neural_net_cosines = neural_nets(input_cosines, output_cosines, expected_cosine)
neural_net_tangents = neural_nets(input_tangents, output_tangents, expected_tangent)


instance_sines = Prediction(input_data, input_sines, output_sines)
instance_cosines = Prediction(input_data, input_cosines, output_cosines)
instance_tangents = Prediction(input_data, input_tangents, output_tangents)

predictions_transformed = (
    instance_sines.execute_transforming,
    instance_cosines.execute_transforming,
    instance_tangents.execute_transforming
)

predictions_normals = (
    instance_sines.execute_normal,
    instance_cosines.execute_normal,
    instance_tangents.execute_normal
)


clear()
print("Model A: `neural_networks.tensorflow`")
print(warning_message, neural_net_sines, neural_net_cosines, neural_net_tangents, sep="\n")

print("Model B: `Prediction` by sklearn")
print(warning_message, predictions_normals, sep="\n")

