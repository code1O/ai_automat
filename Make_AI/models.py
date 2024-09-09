import os, subprocess
import pandas as pd
import numpy as np
import tensorflow as tf
import torch
from .prepare_funcs import (neural_networks, HardPredict, _TypeNum)
from rich.console import Console


def globalReadJson(categorie: str, subcategorie: str, key:str):
    df = pd.read_json("data/machle_data.json")
    get_cat = df.get(categorie)
    subcat = get_cat[0][subcategorie]
    result = subcat[0][key]
    return result

rich_console = Console()

def export_model(format: str = "keras",*, name_file: str, from_folder: str):
    """
    Export any model for implement it with JavaScript/TypeScript
    in the website
    """
    exists_folder = True if os.path.exists(from_folder) else False
    os.chdir(from_folder) if exists_folder else None
    
    args_call = ["tensorflowjs_converter", "--input_format", format, name_file, from_folder]
    
    with rich_console.status("Exporting model...", spinner="dots"):
        subprocess.call(args_call, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    rich_console.print(":white_checkmark: model exported succesfully")


class ReadyUpModel:
    def __init__(self, save_where: str, save_as: str="model",*, category: str, sub_category: str, input_key:str, output_key:str, valueInput: _TypeNum, epochs: int=60):
        self.epochs = epochs
        self.save_as, self.save_where = save_as, save_where
        self.category, self.subcategory = category, sub_category
        self.valueInput = valueInput
        self.input_key, self.output_key = input_key, output_key
    
    def __readJson(self):
        Input_values = globalReadJson(self.category, self.subcategory, self.input_key)
        Output_values = globalReadJson(self.category, self.subcategory, self.output_key)
        X_train = np.array(Input_values, dtype=float)
        Y_train = np.array(Output_values, dtype=float)
        input_units = 16
        return [X_train, Y_train, input_units]
    
    def __prepareHard(self):
        X_train, Y_train, _ = self.__readJson()
        batch_size =  32
        height, width = 224, 224
        needData = dict(
            group_shape=(3,3), matrix_shape=(2,2),
            units=100, inputCat=X_train, outputCat=Y_train,
            size_batch=batch_size, rounds=self.epochs,
            height_=height, width_=width
        )
        return needData

    def predictionHard(self):
        input_data = np.array(self.valueInput)
        needData = self.__prepareHard()
        model = HardPredict(
            height=needData["height_"], width=needData["width_"],
            group_shape=needData["group_shape"], matrix_shape=needData["matrix_shape"],
            units=needData["units"],
            X_train=needData["inputCat"], Y_train=needData["outputCat"],
            size_batch=needData["size_batch"], epochs=needData["rounds"],
            save_as=self.save_as, save_where=self.save_where
        )
        return model.predict(input_data.tobytes)

    def predictSoft(self):
        input_data = np.array(self.valueInput, dtype=float)
        X_train, Y_train, input_units = self.__readJson()
        network = neural_networks(X_train, Y_train)
        tf_network = network.tensor_flow(
            self.save_where,
            self.save_as,
            rounds=self.epochs,
            input_data=input_data, input_shape=input_data.shape,
            input_units=input_units
        )
        return tf_network["result"]

def model_conversion(save_where: str, save_as: str, type_conversion:str, from_unit: str, to_unit: str, value_convert: _TypeNum):
    instance = ReadyUpModel(
        category="Unit_conversions", sub_category=type_conversion,
        input_key=from_unit, output_key=to_unit,
        valueInput=value_convert,
        save_as=save_as, save_where=save_where,
        epochs=400
    )
    soft_conversion = instance.predictSoft()
    adjust_soft_conv = np.array(soft_conversion[0])
    magic_num = np.array([9 * 10])
    soft_conv_tensor = torch.from_numpy(adjust_soft_conv)
    magic_tensor = torch.from_numpy(magic_num)
    aprox_A = soft_conv_tensor - magic_tensor
    
    return aprox_A
