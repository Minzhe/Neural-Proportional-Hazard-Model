import pandas as pd
from pkg_resources import resource_filename

def load_dataset(name, **kwargs):
    '''
    Load a dataset from NeuralPHSurvival.datasets
    Parameters:
    filename : for example "larynx.csv"
    usecols : list of columns in file to use
    Returns : Pandas dataframe
    '''
    return pd.read_csv(resource_filename('NeuralPHSurvival', 'datasets/' + name + '.csv'), engine='python', **kwargs)