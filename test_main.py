import os
import typer
from titanic_model_creator import TitanicModelCreator
from loaders.passenger_loader import PassengerLoader
from loaders.test_loader import TestLoader
from loaders.sql_loader import SqlLoader
from savers.test_model_saver import TestModelSaver
from main import RARE_TITLES
from titanic_model import TitanicModel
from sklearn.linear_model import LogisticRegression
from titanic_model import CategoricalFeature
from titanic_model import NumericalFeature


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
            targets_extractor=lambda passenger: passenger.is_survived,
            features=[
                CategoricalFeature(extractor=lambda passenger: passenger.embarked),
                CategoricalFeature(extractor=lambda passenger: passenger.sex),
                CategoricalFeature(extractor=lambda passenger: passenger.pclass),
                CategoricalFeature(extractor=lambda passenger: passenger.title),
                CategoricalFeature(extractor=lambda passenger: passenger.is_alone),
                NumericalFeature(
                    extractor=lambda passenger: [passenger.age, passenger.fare],
                    n_neighbors=5
                ),
                NumericalFeature(
                    extractor=lambda passenger: [passenger.family_size],
                    n_neighbors=5
                )
            ],
            predictor=LogisticRegression(random_state=0)
        ),
        model_saver=TestModelSaver(data_directory=data_directory)
    )
    titanicModelCreator.run()


if __name__ == "__main__":
    typer.run(test_main)
