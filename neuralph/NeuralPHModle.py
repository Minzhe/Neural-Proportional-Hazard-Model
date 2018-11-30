#####################################################################################
###                             NeuralPHFitter.py                                 ###
#####################################################################################
# Neural network based proportional hazard model

import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import utils

class NeuralPHFitter(object):
    '''
    This class implements fitting neural network based proportional hazard model:
    h(t|x) = h_0(t) * f(x'*beta)

    Parameters:
      alpha: the level in the confidence intervals.
      tie_method: specify how the fitter should deal with ties. Currently only 'Efron' is available.
      strata: specify a list of columns to use in stratification. This is useful if a
         catagorical covariate does not obey the proportional hazard assumption. This
         is used similar to the `strata` expression in R.
         See http://courses.washington.edu/b515/l17.pdf.
    '''

    def __init__(self, alpha=0.95, tie_method='Efron', strata=None):
        if not (0 < alpha <= 1.):
            raise ValueError('alpha parameter must be between 0 and 1.')
        if tie_method != 'Efron':
            raise NotImplementedError("Only Efron is available atm.")
        
        self.alpha = alpha
        self.tie_method = tie_method
        self.strata = strata

    
    def fit(self, df, duration_col, event_col=None, hidden_layer_sizes=None, activation='linear', optimizer='adam', lr=0.001, verbose=False, model_save_path, strata=None):
        '''
        Fit the neural network based proportional hazard model.

        Parameters:
            df: a Pandas dataframe with necessary columns `duration_col` and
                `event_col`, plus other covariates. `duration_col` refers to
                the lifetimes of the subjects. `event_col` refers to whether
                the 'death' events was observed: 1 if observed, 0 else (censored).
            duration_col: the column in dataframe that contains the subjects'
                lifetimes.
            event_col: the column in dataframe that contains the subjects' death
                observation. If left as None, assume all individuals are non-censored.
            hidden_layer_sizes: the network configuration for hidden layers. Tuple or list 
                like objects specify the number of neurons in each hidden layer.
                Default is `None`, no hidden layers, direct to output.
            activation: activation function for the hidden layer
        
            verbose: since the fitter is iterative, show convergence
                diagnostics.
            strata: specify a list of columns to use in stratification. This is useful if a
                catagorical covariate does not obey the proportional hazard assumption. This
                is used similar to the `strata` expression in R.
                See http://courses.washington.edu/b515/l17.pdf.
        
        Returns:
            self, with additional properties: hazards_
        '''
        
        df = df.copy()
        df = df.sort_values(by=duration_col)

        self._n_examples = df.shape[0]
        self.strata = utils.coalesce(strata, self.strata)
        if self.strata is not None:
            original_index = df.index.copy()
            df = df.set_index(self.strata)

        # extract time and event
        T = df[duration_col]
        del df[duration_col]
        if event_col is None:
            E = pd.Series(np.ones(df.shape[0]), index=df.index)
        else:
            E = df[event_col]
            del df[event_col]
        self.data = df
        self._check_values(df, T, E)
        df = df.astype(float)

        # save fitting data for later
        self.durations = T.copy()
        self.event_observed = E.copy()
        if self.strata is not None:
            self.durations.index = original_index
            self.event_observed.index = original_index
        self.event_observed = self.event_observed.astype(bool)

        # normalize
        X = scale(df)

        E = E.astype(bool)

        # fit neural network model
        self._fit_nn(X, T, E, hidden_layer_sizes, activation, optimizer, lr, model_save_path, verbose)

    def _fit_nn(self, X, T, E, hidden_layers, activation, optimizer, lr, model_save_path, verbose=1):
        '''
        Fit neural network model in Keras implementation.

        Note that data is assumed to be sorted on T!

        Parameters:
            X: (n,d) Pandas DataFrame of observations.
            T: (n) Pandas Series representing observed durations.
            E: (n) Pandas Series representing death events.
            hidden_layers: 
            activation: 
            optimizer: 
            lr: learning rate
            verbose:.

        Returns:
            beta: (1,d) numpy array.
        '''

        # model construction
        print('Initilizing neural network model ... ', end='', flush=True)
        n, d = X.shape
        inputs = Input(shape=(n,d))
        for neurons in hidden_layers:
            inputs = Dense(neurons, activation=activation) (inputs)
        output = Dense(1, activation=activation) (inputs)

        model = Model(inputs=[inputs], outputs=[output])
        model.compile(optimizer=optimizer, 
                      loss=lambda x, y: _neg_log_partial_likelihood(x, y[0], y[1]), 
                      metrics=[_log_partial_likelihood, _c_index])
        print('Done\nModel structure summary:', flush=True)
        print(model.summary())

        # model fitting
        print('Start training neural network model ... ', flush=True)
        early_stopper = EarlyStopping(patience=10, verbose=1)
        check_pointer = ModelCheckpoint(model_save_path, verbose=1, save_best_only=True)
        trace = model.fit(X, (T,E), 
                          validation_split=0,
                          batch_size=None,
                          epochs=200,
                          verbose=verbose,
                          shuffle=False,
                          callbacks=[early_stopper, check_pointer])
        self._model = load_model(model_save_path)
        print('Done training')

        return trace
        
    def predict_partial_hazard(self, X):
        '''
        X: a (n,d) covariate numpy array or DataFrame. If a DataFrame, columns
            can be in any order. If a numpy array, columns must be in the
            same order as the training data.

        Returns: the partial hazard for the individuals, partial since the
            baseline hazard is not included. Output of neural network, equal 
            to f(X).
        '''
        if isinstance(X, pd.DataFrame):
            order = self.hazards_.columns
            X = X[order]
        return self._model.predcit(X)
    
    ##########################    internal functions   ##############################
    def _neg_log_partial_likelihood(hr, T, E):
        '''
        Compute the log partial likelihood of the predicted hazard ratio.
        '''
        # sort by time
        idx = np.argsort(T)
        hr, T, E = hr[idx], T[idx], E[idx]
        
        # calculate partial likelihood
        loss = 0
        for i in range(len(E)):
            if E[i] == 1:
                loss += - hr[i] + np.log(np.sum(np.exp(hr[i:])))
        return loss

    




    def _compute_baseline_hazard(self, data, durations, event_observed, name):
        '''
        Compute the baseline hazard
        https://stats.stackexchange.com/questions/46532/cox-baseline-hazard
        '''
        ind_hazards = self.predict_partial_hazard(data)
        ind_hazards['event_at'] = durations.values
        ind_hazards_summed_over_durations = ind_hazards.groupby('event_at')[0].sum().sort_index(ascending=False).cumsum()
        ind_hazards_summed_over_durations.name = 'hazards'

        event_table = survival_table_from_events(durations, event_observed)
        event_table = event_table.join(ind_hazards_summed_over_durations)
        baseline_hazard = pd.DataFrame(event_table['observed'] / event_table['hazards'], columns=[name]).fillna(0)
        return baseline_hazard
    
    @staticmethod
    def _check_values(df, T, E):
        utils.pass_for_numeric_dtypes_or_raise(df)
        utils.check_nans(T)
        utils.check_nans(E)
