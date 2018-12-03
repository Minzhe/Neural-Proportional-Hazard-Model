import pandas as pd
from pkg_resources import resource_filename

def load_dataset(name, processed=False, **kwargs):
    '''
    Load a dataset from NeuralPHSurvival.datasets
    Parameters:
    filename : for example "larynx.csv"
    usecols : list of columns in file to use
    Returns : Pandas dataframe
    '''
    if processed:
        return pd.read_csv(resource_filename('neuralph', 'datasets/' + name + '_processed.csv'), engine='python', **kwargs)
    else:
        return pd.read_csv(resource_filename('neuralph', 'datasets/' + name + '.csv'), engine='python', **kwargs)