import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import KNNImputer


class TitanicModel:

    def __init__(self, n_neighbors=5, predictor=None):
        if predictor is None:
            predictor = LogisticRegression(random_state=0)
        self.trained = False
        self.oneHotEncoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
        self.knnImputer = KNNImputer(n_neighbors=n_neighbors)
        self.robustScaler = RobustScaler()
        self.predictor = predictor

    def process_inputs(self, passengers):
        data = pd.DataFrame([passenger.dict() for passenger in passengers])
        categorical_data = data[['embarked', 'sex', 'pclass', 'title', 'is_alone']]
        numerical_data = data[['age', 'fare', 'family_size']]
        if self.trained:
            categorical_data = self.oneHotEncoder.transform(categorical_data)
            numerical_data = self.robustScaler.transform(self.knnImputer.transform(numerical_data))
        else:
            categorical_data = self.oneHotEncoder.fit_transform(categorical_data)
            numerical_data = self.robustScaler.fit_transform(self.knnImputer.fit_transform(numerical_data))
        return np.hstack((categorical_data, numerical_data))

    def train(self, passengers):
        targets = [passenger.is_survived for passenger in passengers]
        inputs = self.process_inputs(passengers)
        self.predictor.fit(inputs, targets)
        self.trained = True

    def estimate(self, passengers):
        inputs = self.process_inputs(passengers)
        return self.predictor.predict(inputs)
