import pandas as pd
from dlhub_sdk import DLHubClient
from dlhub_sdk.models.servables.python import PythonStaticMethodModel
from mastml.dlhub_predictor import run_dlhub_prediction
import os
import mastml
import shutil

class DLHubHosting():
    '''


    '''
    def __init__(self, Xtrain_path, model_path, preprocessor_path):
        self.Xtrain_path = Xtrain_path
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        return

    def get_input_columns(self):
        X_train = pd.read_excel(os.path.abspath(os.path.join(self.Xtrain_path, 'X_train.xlsx')))
        input_columns = X_train.columns.to_list()
        return input_columns

    def host_model(self, model_title, model_name, model_type="scikit-learn"):
        # Assume the model will be hosted with a list of the input column names
        # input_columns
        # Also assume the model will be hosted with the needed preprocessing routine
        # scaler_path

        dl = DLHubClient()
        if model_type == 'scikit-learn':
            model = PythonStaticMethodModel.from_function_pointer(run_dlhub_prediction)
        else:
            raise ValueError("Only scikit-learn models supported at this time")

        # Some model descriptive info
        model.set_name(model_name).set_title(model_title)

        # Describe the inputs/outputs
        model.set_inputs('list', 'list of material compositions to predict', item_type='string')
        model.set_outputs(data_type='float', description='Predicted value from trained sklearn model')

        # Add additional files to model servable- needed to do featurization of predictions using DLHub
        print('Submitting preprocessor file to DLHub:')
        print(os.path.abspath(self.preprocessor_path))
        print('Submitting model file to DLHub:')
        print(os.path.abspath(self.model_path))
        print('Submitting training data file to DLHub:')
        print(os.path.abspath(self.Xtrain_path))
        print('Submitting mastml directory to DLHub:')
        print(os.path.join(os.path.abspath(mastml.__path__[0])))

        # Need to change model, preprocessor names to be standard model.pkl and preprocessor.pkl names. Copy them and change names
        shutil.copy(os.path.abspath(self.model_path), os.path.join(os.getcwd(), 'model.pkl'))
        shutil.copy(os.path.abspath(self.preprocessor_path), os.path.join(os.getcwd(), 'preprocessor.pkl'))
        shutil.copy(os.path.abspath(self.Xtrain_path), os.path.join(os.getcwd(), 'X_train.csv'))

        model.add_directory(os.path.join(os.path.abspath(mastml.__path__[0])), recursive=True)
        model.add_file('model.pkl')
        model.add_file('preprocessor.pkl')
        model.add_file('X_train.csv')

        # Add pip installable dependency for MAST-ML
        #model.add_requirement('mastml', 'latest')
        model.add_requirement('numpy', '1.18.4')
        model.add_requirement('pandas', '1.1.5')
        model.add_requirement('scikit-learn', '0.23.1')
        model.add_requirement('pymatgen', '2020.1.10')

        res = dl.publish_servable(model)
        return dl, res



