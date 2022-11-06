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
