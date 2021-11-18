# Note that this is work in progress and some hard-coded values were used for initial examples only and will be
# removed (and generalized) in the future

from mastml.feature_generators import ElementalFeatureGenerator
import pandas as pd
import joblib
import os

def featurize_mastml(prediction_input, preprocessor, X_train):
    '''
    prediction_data: a string, list of strings, or path to an excel file to read in compositions to predict
    file_path (str): file path to test data set to featurize
    composition_column_name (str): name of column in test data containing material compositions. Just assume it is 'composition'
    scaler: sklearn normalizer, e.g. StandardScaler() object, fit to the training data
    training_data_path (str): file path to training data set used in original model fit
    '''

    # Write featurizer that takes chemical formula of test materials (from file), constructs correct feature vector then reports predictions

    if type(prediction_input) is str:
        if '.xlsx' in prediction_input:
            df_new = pd.read_excel(prediction_input, header=0)
            df_new.columns = ['composition']
            compositions = df_new['composition'].tolist()
        elif '.csv' in prediction_input:
            df_new = pd.read_csv(prediction_input, header=0)
            df_new.columns = ['composition']
            compositions = df_new['composition'].tolist()
        else:
            compositions = [prediction_input]
            df_new = pd.DataFrame().from_dict(data={'composition': compositions})
    elif type(prediction_input) is list:
        compositions = prediction_input
        df_new = pd.DataFrame().from_dict(data={'composition': compositions})
    else:
        raise TypeError('prediction_data must be a composition in the form of a string, list of strings, or .csv or .xlsx file path')

    X_generated, _ = ElementalFeatureGenerator(composition_df=df_new,
                                               feature_types=['composition_avg', 'arithmetic_avg', 'max', 'min', 'difference', 'element']).fit_transform(X=df_new)

    X_test = X_generated[X_train.columns.to_list()]

    X_test = preprocessor.transform(X_test)

    return X_test

def make_prediction(model, prediction_input, preprocessor_path, X_train_path):
    """
    dlhub_servable : a DLHubClient servable model, used to call DLHub to use cloud resources to run model predictions
    compositions (list) : list of composition strings for data points to predict
    X_test (np array) : array of test X feature matrix
    """

    # Featurize the prediction data
    X_test = featurize_mastml(prediction_input, preprocessor_path, X_train_path)

    y_pred_new = model.predict(X_test)
    pred_dict = dict()
    for comp, pred in zip(prediction_input, y_pred_new.tolist()):
        pred_dict[comp] = pred

    # Save new predictions to excel file in cwd
    df_pred = pd.DataFrame.from_dict(pred_dict, orient='index', columns=['Predicted value'])
    df_pred.to_excel(os.path.join(os.getcwd(),'new_predictions.xlsx'))
    return pred_dict

def run_prediction(prediction_input):
    # comp_list (list): list of strings of material compositions to featurize and predict

    # Note: this function is meant to run in a DLHub container that will have access to the following files:
    #  model.pkl : a trained sklearn model
    #  selected.csv : csv file containing training data
    #  preprocessor.pkl : a preprocessor from sklearn

    # For now, assume we are running from job made on Google Colab. Files stored at /content/filename
    # Load scaler:
    preprocessor = joblib.load('preprocessor.pkl')
    #try:
    #    preprocessor = joblib.load('content/preprocessor.pkl')
    #except FileNotFoundError:
    #    preprocessor = joblib.load('preprocessor.pkl')

    # Load model:
    model = joblib.load('model.pkl')
    #try:
    #    model = joblib.load('content/model.pkl')
    #except FileNotFoundError:
    #    model = joblib.load('model.pkl')

    # Load training data:
    #X_train = pd.read_excel('X_train.xlsx')
    X_train = pd.read_csv('X_train.csv')
    #try:
    #    X_train = pd.read_excel('content/X_train.xlsx')
    #except FileNotFoundError:
    #    X_train = pd.read_excel('X_train.xlsx')

    # TODO: change
    pred_dict = make_prediction(model, prediction_input, preprocessor, X_train)
    # pred_dict = {"hello": "world"}
    return pred_dict