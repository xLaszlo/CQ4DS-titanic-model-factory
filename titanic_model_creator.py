import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


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
