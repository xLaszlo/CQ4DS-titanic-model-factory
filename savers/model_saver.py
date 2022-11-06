import pickle


class ModelSaver:

    def __init__(self, model_filename, result_filename):
        self.model_filename = model_filename
        self.result_filename = result_filename

    def save_model(self, model, result):
        pickle.dump(model, open(self.model_filename, 'wb'))
        pickle.dump(result, open(self.result_filename, 'wb'))
