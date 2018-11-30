#####################################################################################
###                                 utils.py                                      ###
#####################################################################################
# utility functions

import warnings
import numpy as np
import pandas as pd


def concordance_index(event_times, predicted_scores, event_observed=None):
    '''
    Calculates the concordance index (C-index) between two series
    of event times. The first is the real survival times from
    the experimental data, and the other is the predicted survival
    times from a model of some kind.
    The concordance index is a value between 0 and 1 where,
    0.5 is the expected result from random predictions,
    1.0 is perfect concordance and,
    0.0 is perfect anti-concordance (multiply predictions with -1 to get 1.0)
    Score is usually 0.6-0.7 for survival models.

    Parameters:
      event_times: a (n,) array of observed survival times.
      predicted_scores: a (n,) array of predicted scores - these could be survival times, or hazards, etc.
      event_observed: a (n,) array of censorship flags, 1 if observed,
                      0 if not. Default None assumes all observed.
    Returns:
      c-index: a value between 0 and 1.
    '''
    event_times = np.array(event_times, dtype=float)
    predicted_scores = np.array(predicted_scores, dtype=float)

    # Allow for (n, 1) or (1, n) arrays
    if event_times.ndim == 2 and (event_times.shape[0] == 1 or event_times.shape[1] == 1):
        # Flatten array
        event_times = event_times.ravel()
    # Allow for (n, 1) or (1, n) arrays
    if (predicted_scores.ndim == 2 and (predicted_scores.shape[0] == 1 or predicted_scores.shape[1] == 1)):
        # Flatten array
        predicted_scores = predicted_scores.ravel()

    if event_times.shape != predicted_scores.shape:
        raise ValueError('Event times and predictions must have the same shape')
    if event_times.ndim != 1:
        raise ValueError('Event times can only be 1-dimensional: (n,)')

    if event_observed is None:
        event_observed = np.ones(event_times.shape[0], dtype=float)
    else:
        if event_observed.shape != event_times.shape:
            raise ValueError('Observed events must be 1-dimensional of same length as event times')
        event_observed = np.array(event_observed, dtype=float).ravel()

    return _c_index(event_times, predicted_scores, event_observed)


def _c_index(event_times, predicted_event_times, event_observed):
    '''
    Compute the concordance index between the predicted hazard ratio and observed event.
    Assumes the data has been verified by neuralph.utils.concordance_index first.
    '''
    def valid_comparison(time_a, time_b, event_a, event_b):
        '''True if times can be compared.'''
        if time_a == time_b:
            # Ties are only informative if exactly one event happened
            return event_a != event_b
        elif event_a and event_b:
            return True
        elif event_a and time_a < time_b:
            return True
        elif event_b and time_b < time_a:
            return True
        else:
            return False
    
    def concordance_value(time_a, time_b, pred_a, pred_b):
        if pred_a == pred_b:
            # Same as random
            return 0.5
        elif pred_a < pred_b:
            return (time_a < time_b) or (time_a == time_b and event_a and not event_b)
        else:  # pred_a > pred_b
            return (time_a > time_b) or (time_a == time_b and not event_a and event_b)

    paircount = 0.0
    csum = 0.0

    for a in range(0, len(event_times)):
        time_a = event_times[a]
        pred_a = predicted_event_times[a]
        event_a = event_observed[a]
        # Don't want to double count
        for b in range(a + 1, len(event_times)):
            time_b = event_times[b]
            pred_b = predicted_event_times[b]
            event_b = event_observed[b]

            if valid_comparison(time_a, time_b, event_a, event_b):
                paircount += 1.0
                csum += concordance_value(time_a, time_b, pred_a, pred_b)

    if paircount == 0:
        raise ZeroDivisionError('No admissable pairs in the dataset.')
    return csum / paircount

def pass_for_numeric_dtypes_or_raise(df):
    nonnumeric_cols = df.select_dtypes(exclude=[np.number, bool]).columns.tolist()
    if len(nonnumeric_cols) > 0:
        raise TypeError('DataFrame contains nonnumeric columns: {}. Try using pandas.get_dummies to convert the non-numeric column(s) to numerical data, or dropping the column(s).'.format(nonnumeric_cols))

def check_nans(array):
    if pd.isnull(array).any():
        raise TypeError("NaNs were detected in the duration_col and/or the event_col")

def coalesce(*args):
    for arg in args:
        if arg is not None:
            return arg
    return None