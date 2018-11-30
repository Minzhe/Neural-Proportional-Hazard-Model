# from lifelines.datasets import load_rossi
# from lifelines import CoxPHFitter, KaplanMeierFitter
# import matplotlib.pyplot as plt

# rossi_dataset = load_rossi()
# # cph = CoxPHFitter()
# # cph.fit(rossi_dataset, duration_col='week', event_col='arrest', show_progress=True)
# # X = rossi_dataset.drop(["week", "arrest"], axis=1)
# # cph.predict_partial_hazard(X)
# # cph.predict_survival_function(X)

# # kmf = KaplanMeierFitter()
# # kmf.fit(durations=rossi_dataset['week'], event_observed=rossi_dataset['arrest'])
# # kmf.plot()
# rossi_dataset.head()

from neuralph.datasets import load_dataset
from neuralph import NeuralPHFitter

lung = load_dataset('lung').iloc[:,1:]
lung['status'] = lung['status'].apply(lambda x: 1 if x == 2 else 0)
lung['ph.ecog'].fillna(value=1, inplace=True)
lung['ph.karno'].fillna(value=lung['ph.karno'].mean(), inplace=True)
lung['pat.karno'].fillna(value=lung['pat.karno'].mean(), inplace=True)
lung['meal.cal'].fillna(value=lung['meal.cal'].mean(), inplace=True)
lung['wt.loss'].fillna(value=lung['wt.loss'].mean(), inplace=True)

nph = NeuralPHFitter()
nph.fit(lung, duration_col='time', event_col='status')