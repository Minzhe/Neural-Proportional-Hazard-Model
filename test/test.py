import sys
sys.path.append('..')
from neuralph.datasets import load_dataset
from neuralph import NeuralPHFitter
from neuralph import utils
from lifelines.datasets import load_rossi
from lifelines import CoxPHFitter, KaplanMeierFitter
import matplotlib.pyplot as plt

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


lung = load_dataset('lung_tcga', index_col=0)
lung.drop(['DSS', 'DSS.time', 'DFI', 'DFI.time', 'PFI', 'PFI.time'], axis=1, inplace=True)
lung.dropna(inplace=True)

nph = NeuralPHFitter(lung, duration_col='OS.time', event_col='OS', hidden_layer_sizes=(64,32), activation='tanh', dropout=0.2)
trace = nph.fit(verbose=1, lr=0.001, epoch=300, optimizer='Adam', validation_split=0.33, model_save_path='lung_tcga.h5', log_dir='log')
# nph.load('nph.h5')
# partial_hazard = nph.predict_partial_hazard()
# nph.plot_baseline()
nph.summary
train_pred = nph.predict_partial_hazard()
plt.show()

# cph = CoxPHFitter()
# cph.fit(lung, duration_col='time', event_col='status')
# cph.print_summary()
# cph.plot_covariate_groups('sex', [1,2])
# plt.show()

# nph = NeuralPHFitter(rossi, duration_col='week', event_col='arrest', hidden_layer_sizes=(16,8), activation='relu')
# trace = nph.fit(verbose=1, lr=0.01, epoch=100, optimizer='Adam', model_save_path='rossi.nph.h5')
# print(nph.summary)

# from neuralph.utils import concordance_index
# import numpy as np
# pred_score = np.array([0.7, 0.5, 0.3, 0.4, 0.2, 0.1, 0.05, 0.2])
# T = np.array([1,2,3,4,5,6,7,8])
# E = np.array([0,1,0,0,1,1,1,1])
# print(concordance_index(pred_score, T, E))