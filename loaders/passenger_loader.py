from datamodel.passenger import Passenger


class PassengerLoader:

    def __init__(self, loader, rare_titles=None):
        self.loader = loader
        self.rare_titles = rare_titles

    def get_passengers(self):
        passengers = []
        for data in self.loader.get_passengers().itertuples():
            # parch = Parents/Children, sibsp = Siblings/Spouses
            family_size = int(data.parch + data.sibsp)
            # Allen, Miss. Elisabeth Walton
            title = data.name.split(',')[1].split('.')[0].strip()
            passenger = Passenger(
                pid=int(data.pid),
                pclass=int(data.pclass),
                sex=str(data.sex),
                age=float(data.age),
                family_size=family_size,
                fare=float(data.fare),
                embarked=str(data.embarked),
                is_alone= 1 if family_size==1 else 0,
                title='rare' if title in self.rare_titles else title,
                is_survived=int(data.is_survived)
            )
            passengers.append(passenger)
        return passengers
