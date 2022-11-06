import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import KNNImputer


class CategoricalFeature:

    def __init__(self, extractor):
        self.extractor = extractor
        self.oneHotEncoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

    def fit_transform(self, passengers):
        data = pd.DataFrame([self.extractor(passenger) for passenger in passengers])
        return self.oneHotEncoder.fit_transform(data)

    def transform(self, passengers):
        data = pd.DataFrame([self.extractor(passenger) for passenger in passengers])
        return self.oneHotEncoder.transform(data)


class NumericalFeature:

    def __init__(self, extractor, n_neighbors=5):
        self.extractor = extractor
        self.knnImputer = KNNImputer(n_neighbors=n_neighbors)
        self.robustScaler = RobustScaler()

    def fit_transform(self, passengers):
        data = pd.DataFrame([self.extractor(passenger) for passenger in passengers])
        return self.robustScaler.fit_transform(self.knnImputer.fit_transform(data))

    def transform(self, passengers):
        data = pd.DataFrame([self.extractor(passenger) for passenger in passengers])
        return self.robustScaler.transform(self.knnImputer.transform(data))


class TitanicModel:

    def __init__(self, features, targets_extractor, predictor):
        self.trained = False
        self.features = features
        self.targets_extractor = targets_extractor
        self.predictor = predictor

    def process_inputs(self, passengers):
        result = None
        for feature in self.features:
            if self.trained:
                data = feature.transform(passengers)
            else:
                data = feature.fit_transform(passengers)
            if result is None:
                result = data
            else:
                result = np.hstack((result, data))
        return result

    def train(self, passengers):
        inputs = self.process_inputs(passengers)
        targets = [self.targets_extractor(passenger) for passenger in passengers]
        self.predictor.fit(inputs, targets)
        self.trained = True

    def estimate(self, passengers):
        inputs = self.process_inputs(passengers)
        return self.predictor.predict(inputs)
