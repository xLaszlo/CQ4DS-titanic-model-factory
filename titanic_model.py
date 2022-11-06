import os
import pickle
import typer
import numpy as np
import pandas as pd

from loaders.passenger_loader import PassengerLoader
from loaders.test_loader import TestLoader
from loaders.sql_loader import SqlLoader
from savers.model_saver import ModelSaver
from savers.test_model_saver import TestModelSaver

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import KNNImputer
from sklearn.metrics import confusion_matrix


RARE_TITLES = {
    'Capt',
    'Col',
    'Don',
    'Dona',
    'Dr',
    'Jonkheer',
    'Lady',
    'Major',
    'Mlle',
    'Mme',
    'Ms',
    'Rev',
    'Sir',
    'the Countess'
}


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


class TitanicModelCreator:

    def __init__(self, loader, model, model_saver):
        self.loader = loader
        self.model = model
        self.model_saver = model_saver
        np.random.seed(42)

    def get_train_pids(self, passengers):
        pids = [passenger.pid for passenger in passengers]
        targets = [passenger.is_survived for passenger in passengers]
        train_pids, test_pids = train_test_split(pids, stratify=targets, test_size=0.2)
        return train_pids, test_pids

    def run(self):
        passengers = self.loader.get_passengers()
        passengers_map = {p.pid: p for p in passengers}
        train_pids, test_pids = self.get_train_pids(passengers)
        train_passengers = [passengers_map[pid] for pid in train_pids]
        test_passengers = [passengers_map[pid] for pid in test_pids]

        # --- TRAINING ---
        self.model.train(train_passengers)

        y_train_estimation = self.model.estimate(train_passengers)
        y_train = [passengers_map[pid].is_survived for pid in train_pids]
        cm_train = confusion_matrix(y_train, y_train_estimation)

        # --- TESTING ---
        y_test_estimation = self.model.estimate(test_passengers)
        y_test = [passengers_map[pid].is_survived for pid in test_pids]
        cm_test = confusion_matrix(y_test, y_test_estimation)

        self.model_saver.save_model(
            model=self.model,
            result={
                'cm_train': cm_train,
                'cm_test': cm_test,
                'train_passengers': train_passengers,
                'test_passengers': test_passengers
            }
        )


def main(data_directory:str):
    titanicModelCreator = TitanicModelCreator(
        loader=PassengerLoader(
            loader=SqlLoader(
                connectionString=f'sqlite:///{data_directory}/titanic.db'
            ),
            rare_titles=RARE_TITLES
        ),
        model=TitanicModel(),
        model_saver=ModelSaver(
            model_filename=f'{data_directory}/real_model.pkl',
            result_filename=f'{data_directory}/real_result.pkl'
        )
    )
    titanicModelCreator.run()


def test_main(data_directory:str):
    database_filename = f'{data_directory}/titanic.db'
    if not os.path.isfile(database_filename):
        raise ValueError(f'Database not found at {database_filename}')
    titanicModelCreator = TitanicModelCreator(
        loader=PassengerLoader(
            loader=TestLoader(
                passengers_filename=f'{data_directory}/passengers.pkl',
                realLoader=SqlLoader(
                    connectionString=f'sqlite:///{database_filename}'
                )
            ),
            rare_titles=RARE_TITLES
        ),
        model=TitanicModel(
            n_neighbors=5,
            predictor=LogisticRegression(random_state=0)
        ),
        model_saver=TestModelSaver(data_directory=data_directory)
    )
    titanicModelCreator.run()


if __name__ == "__main__":
    typer.run(test_main)
