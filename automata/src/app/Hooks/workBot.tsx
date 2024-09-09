import * as tf from "@tensorflow/tfjs";
import React, { useEffect, useState } from "react";
import { PythonShell } from "python-shell";
require("process")


export function ModelPredict(modelPath: string, valuePredict: number){
    const [model, setModel] = useState<tf.LayersModel | null>(null);
    const [prediction, setPrediction] = useState<number | null>(null);
    const tensor = tf.tensor([[valuePredict]]);

    useEffect(() => {
        const loadModel = async () => {
            const loadedModel = await tf.loadLayersModel(modelPath);
            setModel(loadedModel);
        };
    }, [])
    const handlePredict = async() => {
        const predictTensor = model?.predict(tensor) as tf.Tensor;
        const predicted = predictTensor.dataSync()[0];
        setPrediction(predicted);
    }
    return (
        <div>
            {/* <Component onEventClick={handlePredict} */}
            {prediction != null && <p>Model predicts: {prediction}</p>}
        </div>
    )
}

export function displayCode() {}


type compileOptions = {
    lang: string,
    code: string
}

export function compileCode({lang, code}: compileOptions) {
    const [ result, setResult ] = useState<any | null>(null);
    const PythonOptions = {
        path: `C:\\312\\python.exe`,
        args: ["value1", "value2"]
    }
}