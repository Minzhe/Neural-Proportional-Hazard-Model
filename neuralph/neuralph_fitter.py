#####################################################################################
###                             NeuralPHFitter.py                                 ###
#####################################################################################
# Neural network based proportional hazard model

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from keras.models import load_model, Sequential
from keras.layers import Dense
from keras.optimizers import Adam, SGD, RMSprop
from keras.activations import relu, tanh, linear
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
import tensorflow as tf
from neuralph import utils

class NeuralPHFitter(object):
    '''
    This class implements fitting neural network based proportional hazard model:
    h(t|x) = h_0(t) * f(x'*beta)

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
        activation: activation function for the hidden layer, default 'linear'
        optimizer: optimizer for training the neural network, default 'adam'
        alpha: the level in the confidence intervals.
        tie_method: specify how the fitter should deal with ties. Currently only 'Efron' is available.
        strata: specify a list of columns to use in stratification. This is useful if a
            catagorical covariate does not obey the proportional hazard assumption. This
            is used similar to the `strata` expression in R.
            See http://courses.washington.edu/b515/l17.pdf.
    '''

    def __init__(self, df, duration_col, event_col=None, hidden_layer_sizes=None, activation='linear', alpha=0.95, tie_method='Efron', strata=None):
        # parameter
        if not (0 < alpha <= 1.):
            raise ValueError('Alpha parameter must be between 0 and 1.')
        if tie_method != 'Efron':
            raise NotImplementedError("Only Efron is available atm.")
        self.alpha = alpha
        self.tie_method = tie_method
        self.strata = strata

        # process data
        df = df.copy()
        df = df.sort_values(by=duration_col)
        # self.strata = utils.coalesce(strata, self.strata)
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

        # save fitting data for later
        self._check_values(df, T, E)
        self.data = df.copy()
        self.n_samples, self.n_features = df.shape
        self._columns = list(df.columns)
        df = df.astype(float)
        self.durations = T.copy()
        self.event_observed = E.copy()
        if self.strata is not None:
            self.durations.index = original_index
            self.event_observed.index = original_index
        self.event_observed = self.event_observed.astype(bool)

        # normalize and transform
        self._X, self._col_mean, self._col_std = self._scale_columns(df)
        self._y = np.array(list(zip(self.durations, self.event_observed)))

        # initialize model parameters
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation


    def fit(self, optimizer='Adam', lr=0.001, epoch=200, batch_size=None, validation_split=0, verbose=False, model_save_path='nph.h5'):
        '''
        Fit the neural network based proportional hazard model.

        Parameters:
            lr: learning rate.
            batch_size: number of samples in each batch, default is None, which is does not divide mini-batches.
            validation_solit: proportion of samples used for validation, default is 0.
            verbose: whether display progress.
            model_save_path: where to store the model, defacult stored in the current directory.
        Returns:
            trace of training process
        '''

        # optimizer
        self.lr = lr
        self.optimizer = optimizer
        if self.optimizer == 'Adam':
            optimizer = Adam(lr=self.lr)
        elif self.optimizer == 'SGD':
            optimizer = SGD(lr=self.lr)
        elif self.optimizer == 'RMSprop':
            optimizer = RMSprop(lr=self.lr)
        else:
            raise ValueError('Unrecognizable optimizer, should be one of the following: Adam, SGD, RMSprop')

        # initilize model
        print('Initilizing neural network model ... ', end='', flush=True)
        model = Sequential()
        if self.hidden_layer_sizes is None:       # default linear regression
            model.add(Dense(1, activation='linear', input_dim=self.n_features))
        else:                                     # build hidden layers
            for i, neurons in enumerate(self.hidden_layer_sizes):
                if i == 0:
                    model.add(Dense(neurons, activation=self.activation, input_dim=self.n_features))
                else:
                    model.add(Dense(neurons, activation=self.activation))
            model.add(Dense(1, activation='linear'))
        
        model.compile(optimizer=optimizer, 
                      loss=self._neg_log_partial_likelihood,
                      metrics=[self._concordance_index])
        print('Done\nModel structure summary:', flush=True)
        print(model.summary())

        # fitting configuration
        if batch_size is None:
            batch_size = self.n_samples
        monitor = 'loss'
        if validation_split > 0:
            if validation_split * self.n_samples < 30:
                raise ValueError('Validation data size too small, consider increase validation_split or set to 0 (use all data for training and validation).')
            monitor = 'val_loss'
        
        # model fitting
        print('Start training neural network model ... ', flush=True)
        early_stopper = EarlyStopping(monitor=monitor, patience=10, verbose=1)
        check_pointer = ModelCheckpoint(model_save_path, monitor=monitor, verbose=1, save_best_only=True)
        trace = model.fit(self._X, self._y,
                                validation_split=validation_split,
                                batch_size=batch_size,
                                epochs=epoch,
                                verbose=verbose,
                                shuffle=False,
                                callbacks=[early_stopper, check_pointer])
        model = load_model(model_save_path, 
                           custom_objects={'_neg_log_partial_likelihood': self._neg_log_partial_likelihood,
                                           '_concordance_index': self._concordance_index})
        self.model = model
        print('\nDone training')
        self.summarize()

        return trace.history
    

    def load(self, path):
        '''
        Load trained model.
        '''
        print('Loading model from {} ...'.format(path))
        self.model = load_model(path, custom_objects={'_neg_log_partial_likelihood': self._neg_log_partial_likelihood,
                                                      '_concordance_index': self._concordance_index})
        self.summarize()
        
    def summarize(self):
        self.variable_hazard = self._compute_variable_hazard()
        self.baseline_hazard, self.baseline_cumulative_hazard, self.baseline_survival = self._compute_baseline_survival()

    @property
    def summary(self):
        '''
        Print the summary of fitted model.
        '''
        print(self.variable_hazard)
        return self.baseline_survival


    def predict_partial_hazard(self, X):
        '''
        X: a (n,d) covariate numpy array or DataFrame. If a DataFrame, columns
            can be in any order. If a numpy array, columns must be in the
            same order as the training data.

        Returns: the partial hazard for the individuals, partial since the
            baseline hazard is not included. Output of neural network, equal 
            to f(X).
        '''
        hazard_names = self._columns
        if isinstance(X, pd.DataFrame):
            order = hazard_names
            X = X[order]
            utils._pass_for_numeric_dtypes_and_no_na(X)
        elif isinstance(X, pd.Series) and ((X.shape[0] == len(hazard_names) + 2) or (X.shape[0] == len(hazard_names))):
            X = X.to_frame().T
            order = hazard_names
            X = X[order]
            utils._pass_for_numeric_dtypes_and_no_na(X)
        elif isinstance(X, pd.Series):
            assert len(hazard_names) == 1, 'Series not the correct arugment'
            X = pd.DataFrame(X)
            utils._pass_for_numeric_dtypes_and_no_na(X)
        
        X = X.astype(float)
        index = utils._get_index(X)

        X = self._scale_columns(X, mean=self._col_mean, std=self._col_std)
        return pd.DataFrame(np.exp(self.model.predict(X)), index=index, columns=['partial_hazard'])
    
    #########################    internal functions   ##############################
    def _compute_variable_hazard(self):
        '''
        Compute the coefficient (log(hazard)) of each variable.
        This try to explain the effect neural network variables in linear form.
        '''
        assert self._X.shape[1] == len(self._columns), 'Columns of X is not equal to feature number'

        pred_full = self.model.predict(self._X)
        hazard = pd.Series(index=self._columns)
        for i in range(len(self._columns)):
            tmp_X = self._X.copy()
            tmp_X[:,i] = 0
            pred_reduce = self.model.predict(tmp_X)
            hazard[i] = np.nanmean((pred_full - pred_reduce) / self._X[:,i]) / self._col_std[i]

        return hazard


    def _compute_baseline_survival(self):
        '''
        Importantly, this agrees with what the Kaplan Meier produces.
        https://stats.stackexchange.com/questions/46532/cox-baseline-hazard
        '''
        ind_hazards = np.exp(self.predict_partial_hazard(self._X))
        ind_hazards['event_at'] = self.durations
        ind_hazards_summed_over_durations = ind_hazards.groupby('event_at')['partial_hazard'].sum().sort_index(ascending=False).cumsum()
        ind_hazards_summed_over_durations.name = 'partial_hazard'

        event_table = utils.survival_table_from_events(self.durations, self.event_observed)
        event_table = event_table.join(ind_hazards_summed_over_durations)
        baseline_hazard = pd.DataFrame(event_table['observed'] / event_table['partial_hazard'], columns=['baseline_hazard']).fillna(0)

        baseline_cumulative_hazard = baseline_hazard.cumsum()
        baseline_cumulative_hazard.columns = ['baseline_cumulative_hazard']

        baseline_survival = np.exp(-baseline_cumulative_hazard)
        baseline_survival.columns = ['baseline_survival']

        return baseline_hazard, baseline_cumulative_hazard, baseline_survival


    # >>>>>>>>>>>>>>>>  tensorflow function  <<<<<<<<<<<<<<<< #
    @staticmethod
    def _neg_log_partial_likelihood(T_E, pred_score):
        '''
        Compute the log partial likelihood of the predicted hazard ratio.
        '''
        T = tf.slice(T_E, [0,0], [-1,1])
        E = tf.slice(T_E, [0,1], [-1,1])

        # sort by time
        idx = tf.contrib.framework.argsort(T, axis=0)
        pred_score = tf.gather(pred_score, idx)
        T = tf.gather(T, idx)
        E = tf.gather(E, idx)
        
        # calculate partial likelihood
        hr = tf.exp(pred_score)
        hr_cumsum = tf.cumsum(hr, reverse=True)
        likeli = tf.divide(hr, hr_cumsum)
        partial_likeli = tf.boolean_mask(likeli, E)
        log_partial_likeli = tf.log(partial_likeli)
        log_partial_likeli_sum = tf.reduce_sum(log_partial_likeli)
        neg_log_partial_likeli_sum = tf.negative(log_partial_likeli_sum)

        return neg_log_partial_likeli_sum

    @staticmethod
    def _concordance_index(T_E, pred_score):
        '''
        Compute the concordance index between the predicted hazard ratio and observed event.
        '''
        T = tf.slice(T_E, [0,0], [-1,1])
        E = tf.slice(T_E, [0,1], [-1,1])

        # sort by time
        idx = tf.contrib.framework.argsort(T, axis=0)
        pred_score = tf.gather(pred_score, idx)
        T = tf.gather(T, idx)
        E = tf.gather(E, idx)

        # compute pairwise matrix
        score_mat = tf.subtract(pred_score, tf.reshape(pred_score, [-1,1]))
        T_mat = tf.subtract(T, tf.reshape(T, [-1,1]))
        E_mat_either = tf.add(E, tf.reshape(E, [-1,1]))
        E_mat_diff = tf.subtract(E, tf.reshape(E, [-1,1]))

        # valid comparsion pair
        valid_comp1 = tf.logical_and(tf.equal(T_mat, 0), tf.equal(E_mat_either, 1))
        valid_comp2 = tf.logical_and(tf.not_equal(T_mat, 0), tf.equal(E_mat_either, 2))
        valid_comp3 = tf.logical_and(tf.equal(E_mat_diff, -1), tf.greater(T_mat, 0))
        valid_comp = tf.logical_or(tf.logical_or(valid_comp1, valid_comp2), valid_comp3)
        valid_comp = tf.linalg.band_part(valid_comp, 0, -1)

        # good prediction
        random = tf.logical_and(tf.equal(score_mat, 0), valid_comp)
        good1 = tf.logical_and(
                    tf.logical_and(
                        tf.greater(score_mat, 0), tf.greater(T_mat, 0)
                    ), valid_comp
                )
        good2 = tf.logical_and(
                    tf.logical_and(
                        tf.logical_and(
                            tf.greater(score_mat, 0), tf.equal(T_mat, 0)
                        ), tf.equal(E_mat_diff, -1)
                    ), valid_comp
                )

        # calculate concordance index
        random = tf.multiply(tf.reduce_sum(tf.to_float(random)), 0.5)
        good = tf.add(
            tf.reduce_sum(tf.to_float(good1)), 
            tf.reduce_sum(tf.to_float(good2))
        )
        pairs = tf.reduce_sum(tf.to_float(valid_comp))
        c_index = tf.divide(tf.add(random, good), pairs)

        return tf.subtract(1.0, c_index)

    @staticmethod
    def _check_values(df, T, E):
        utils._pass_for_numeric_dtypes_and_no_na(df)
        utils._check_nans(T)
        utils._check_nans(E)
    
    @staticmethod
    def _scale_columns(df, mean=None, std=None):
        if mean is None and std is None:
            col_means = df.mean(axis=0)
            col_std = df.std(axis=0, ddof=1)
            return np.array((df - col_means) / col_std), np.array(col_means), np.array(col_std)
        elif mean is not None and std is not None:
            return (df - mean) / std
        else:
            raise ValueError('mean and std should be both provided or both None')
