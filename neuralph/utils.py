#####################################################################################
###                                 utils.py                                      ###
#####################################################################################
# utility functions

import warnings
import numpy as np
import pandas as pd

__all__ = ['concordance_index']

######################   calculate concordance   ########################
def concordance_index(predicted_scores, event_times, event_observed=None):
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

    def _valid_comparison(time_a, time_b, event_a, event_b):
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

    def _concordance_value(time_a, time_b, pred_a, pred_b, event_a, event_b):
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
            if _valid_comparison(time_a, time_b, event_a, event_b):
                paircount += 1.0
                csum += _concordance_value(time_a, time_b, pred_a, pred_b, event_a, event_b)

    if paircount == 0:
        raise ZeroDivisionError('No admissable pairs in the dataset.')
    return csum / paircount


def survival_table_from_events(death_times, event_observed, birth_times=None,
                               columns=['removed', 'observed', 'censored', 'entrance', 'at_risk'],
                               collapse=False, intervals=None):
    '''
    Parameters:
        death_times: (n,) array of event times
        event_observed: (n,) boolean array, 1 if observed event, 0 is censored event.
        birth_times: a (n,) array of numbers representing
          when the subject was first observed. A subject's death event is then at [birth times + duration observed].
          If None (default), birth_times are set to be the first observation or 0, which ever is smaller.
        columns: a 3-length array to call the, in order, removed individuals, observed deaths
          and censorships.
        collapse: Default False. If True, collapses survival table into lifetable to show events in interval bins
        intervals: Default None, otherwise a list/(n,1) array of interval edge measures. If left as None
          while collapse=True, then Freedman-Diaconis rule for histogram bins will be used to determine intervals.
    Returns:
        Pandas DataFrame with index as the unique times or intervals in event_times. The columns named
        'removed' refers to the number of individuals who were removed from the population
        by the end of the period. The column 'observed' refers to the number of removed
        individuals who were observed to have died (i.e. not censored.) The column
        'censored' is defined as 'removed' - 'observed' (the number of individuals who
         left the population due to event_observed)
    Example:
        Uncollapsed
                  removed  observed  censored  entrance   at_risk
        event_at
        0               0         0         0        11        11
        6               1         1         0         0        11
        7               2         2         0         0        10
        9               3         3         0         0         8
        13              3         3         0         0         5
        15              2         2         0         0         2
        Collapsed
                 removed observed censored at_risk
                     sum      sum      sum     max
        event_at
        (0, 2]        34       33        1     312
        (2, 4]        84       42       42     278
        (4, 6]        64       17       47     194
        (6, 8]        63       16       47     130
        (8, 10]       35       12       23      67
        (10, 12]      24        5       19      32
    '''
    removed, observed, censored, entrance, at_risk = columns
    death_times = np.asarray(death_times)
    if birth_times is None:
        birth_times = min(0, death_times.min()) * np.ones(death_times.shape[0])
    else:
        birth_times = np.asarray(birth_times)
        if np.any(birth_times > death_times):
            raise ValueError('birth time must be less than time of death.')

    # deal with deaths and censorships
    df = pd.DataFrame(death_times, columns=['event_at'])
    df[removed] = np.asarray(1)
    df[observed] = np.asarray(event_observed).astype(bool)
    death_table = df.groupby('event_at').sum()
    death_table[censored] = (death_table[removed] - death_table[observed]).astype(int)

    # deal with late births
    births = pd.DataFrame(birth_times, columns=['event_at'])
    births[entrance] = np.asarray(1)
    births_table = births.groupby('event_at').sum()
    event_table = death_table.join(births_table, how='outer', sort=True).fillna(0)  # http://wesmckinney.com/blog/?p=414
    event_table[at_risk] = event_table[entrance].cumsum() - event_table[removed].cumsum().shift(1).fillna(0)

    # group by intervals
    if collapse:
        event_table = _group_event_table_by_intervals(event_table, intervals)

    return event_table

def _group_event_table_by_intervals(event_table, intervals):
    event_table = event_table.reset_index()
    # use Freedman-Diaconis rule to determine bin size if user doesn't define intervals
    if intervals is None:
        event_max = event_table['event_at'].max()
        # need interquartile range for bin width
        q75, q25 = np.percentile(event_table['event_at'], [75, 25])
        event_iqr = q75 - q25
        bin_width = 2 * event_iqr * (len(event_table['event_at']) ** (-1 / 3))
        intervals = np.arange(0, event_max + bin_width, bin_width)
    return event_table.groupby(pd.cut(event_table['event_at'], intervals)).agg({'removed': ['sum'],
                                                                                'observed': ['sum'],
                                                                                'censored': ['sum'],
                                                                                'at_risk': ['max']})

######################   utility function   ########################
def _pass_for_numeric_dtypes_and_no_na(df):
    nonnumeric_cols = df.select_dtypes(exclude=[np.number, bool]).columns.tolist()
    if len(nonnumeric_cols) > 0:
        raise TypeError('DataFrame contains nonnumeric columns: {}. Try using pandas.get_dummies to convert the non-numeric column(s) to numerical data, or dropping the column(s).'.format(nonnumeric_cols))
    na_cols = [col for col in df.columns.values if df[col].isnull().any()]
    if len(na_cols) > 0:
        raise ValueError('DataFrame columns {} contains NaN.'.format(na_cols))

def _check_nans(array):
    if pd.isnull(array).any():
        raise TypeError("NaNs were detected in the duration_col and/or the event_col")

def _coalesce(*args):
    for arg in args:
        if arg is not None:
            return arg
    return None

def _get_index(X):
    # we need a unique index because these are about to become column names.
    if isinstance(X, pd.DataFrame) and X.index.is_unique:
        index = list(X.index)
    else:
        # If it's not a dataframe, order is up to user
        index = list(range(X.shape[0]))
    return index