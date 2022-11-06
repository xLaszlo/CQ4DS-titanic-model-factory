import os
import pickle
import pandas as pd


def do_test(filename, data):
    if not os.path.isfile(filename):
        pickle.dump(data, open(filename, 'wb'))
    truth = pickle.load(open(filename, 'rb'))
    try:
        np.testing.assert_almost_equal(data, truth)
        print(f'{filename} test passed')
    except AssertionError as ex:
        print(f'{filename} test failed {ex}')


def do_pandas_test(filename, data):
    if not os.path.isfile(filename):
        data.to_pickle(filename)
    truth = pd.read_pickle(filename)
    try:
        pd.testing.assert_frame_equal(data, truth)
        print(f'{filename} pandas test passed')
    except AssertionError as ex:
        print(f'{filename} pandas test failed {ex}')


class TestModelSaver:

    def __init__(self, data_directory):
        self.data_directory = data_directory

    def save_model(self, model, result):
        do_test(f'{self.data_directory}/cm_test.pkl', result['cm_test'])
        do_test(f'{self.data_directory}/cm_train.pkl', result['cm_train'])
        X_train_processed = model.process_inputs(result['train_passengers'])
        do_test(f'{self.data_directory}/X_train_processed.pkl', X_train_processed)
        X_test_processed = model.process_inputs(result['test_passengers'])
        do_test(f'{self.data_directory}/X_test_processed.pkl', X_test_processed)
        X_train = pd.DataFrame([v.dict() for v in result['train_passengers']])
        do_pandas_test(f'{self.data_directory}/X_train.pkl', X_train)
