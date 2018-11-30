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

from NeuralPHSurvival.datasets import load_dataset
lung = load_dataset('lung')
print(lung)