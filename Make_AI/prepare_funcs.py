import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler

scale = StandardScaler()
poly_regress = lambda x, y, n: np.poly1d(np.polyfit(x,y,n))

class prediction:
    def __init__(self, csv_file, categories, predict_categorie) -> None:
        self.csv = csv_file
        self.categories, self.predict_categorie = categories, predict_categorie
    
    @property
    def initialize_normal(self):
        df = pd.read_csv(self.csv)
        X = df[self.categories]
        y = df[self.predict_categorie]
        regr = linear_model.LinearRegression()
        return regr.fit(X, y)
    
    def execute_normal(self, values):
        instance = self.initialize_normal
        coeficient, prediction = instance.coef_, instance.predict([values])
        dictionary_results = dict(coef=coeficient, predict=prediction)
        return dictionary_results

    @property
    def initialize_transform(self):
        df = pd.read_csv(self.csv)
        X = df[self.categories]
        y = df[self.predict_categorie]
        scaledX = scale.fit_transform(X)
        
        regr = linear_model.LinearRegression()
        return regr.fit(scaledX, y)
    
    def execute_transform(self, values):
        instance = self.initialize_transform
        scaled = scale.transform([values])
        prediction = instance.predict([scaled[0]])
        return prediction
    