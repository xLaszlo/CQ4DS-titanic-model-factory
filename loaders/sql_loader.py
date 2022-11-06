import pandas as pd
from sqlalchemy import create_engine


class SqlLoader:

    def __init__(self, connectionString):
        engine = create_engine(connectionString)
        self.connection = engine.connect()

    def get_passengers(self):
        query = """
            SELECT
                tbl_passengers.pid,
                tbl_passengers.pclass,
                tbl_passengers.sex,
                tbl_passengers.age,
                tbl_passengers.parch,
                tbl_passengers.sibsp,
                tbl_passengers.fare,
                tbl_passengers.embarked,
                tbl_passengers.name,
                tbl_targets.is_survived
            FROM
                tbl_passengers
            JOIN
                tbl_targets
            ON
                tbl_passengers.pid=tbl_targets.pid
        """
        return pd.read_sql(query, con=self.connection)
