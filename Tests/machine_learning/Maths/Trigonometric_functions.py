import os, sys

import torchtext
torchtext.disable_torchtext_deprecation_warning()

import numpy as np
import pandas

# ~/Make_AI`
from Make_AI import neural_networks


clear = lambda: os.system("cls")

df = pandas.read_json("data/machle_data.json")
Trigonometric = df.get("Trigonometric")
sines = Trigonometric[0]["sines"]
cosines = Trigonometric[0]["cosines"]
tangents = Trigonometric[0]["tangents"]

warning_message = "Warning: the predicted results could be round"

input_sines, output_sines = (
    np.array(sines[0]["input"]),
    np.array(sines[0]["output"])
)

input_cosines, output_cosines = (
    np.array(cosines[0]["input"]),
    np.array(cosines[0]["output"])
)

input_tangents, output_tangents = (
    np.array(tangents[0]["input"]),
    np.array(tangents[0]["output"])
)

input_shape, input_units = (1,), 15
input_data = np.array([153.0])

def neural_nets(input_categorie, output_categorie):
    neural_net = neural_networks(input_categorie, output_categorie)
    tf_neural_net = neural_net.tensor_flow(
        rounds=60, optimizer_value=1.0,
        input_data=input_data, input_shape=input_shape,
        input_units=input_units
    )
    tf_result = tf_neural_net["result"]
    return tf_result

neural_net_sines = neural_nets(input_sines, output_sines)
neural_net_cosines = neural_nets(input_cosines, output_cosines)

clear()
print(warning_message, neural_net_cosines, neural_net_sines, sep="\n")
