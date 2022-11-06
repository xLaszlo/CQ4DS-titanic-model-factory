import typer
from titanic_model_creator import TitanicModelCreator
from loaders.passenger_loader import PassengerLoader
from loaders.sql_loader import SqlLoader
from savers.model_saver import ModelSaver
from titanic_model import TitanicModel


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


if __name__ == "__main__":
    typer.run(main)
