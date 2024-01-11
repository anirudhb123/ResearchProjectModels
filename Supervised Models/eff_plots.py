# Imports basics

import numpy as np
import pandas as pd
import h5py
import keras.backend as K
import tensorflow as tf
import json
import random
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_curve

import matplotlib
import matplotlib.pyplot as plt

# best model
tpr1 = np.load('/abvol/bins/(vbfhh,supervised).npy')
fpr1 = np.load('/abvol/divisors/(vbfhh,supervised).npy')

tpr2 = np.load('/abvol/bins/(ztt,supervised).npy')
fpr2 = np.load('/abvol/divisors/(ztt,supervised).npy')

tpr3 = np.load('/abvol/bins/(zvv,supervised).npy')
fpr3 = np.load('/abvol/divisors/(zvv,supervised).npy')

tpr4 = np.load('/abvol/bins/(ttbar,supervised).npy')
fpr4 = np.load('/abvol/divisors/(ttbar,supervised).npy')



# Define colors and line styles for each group
colors = ['blue', 'orange', 'green', 'red']

# Create a new figure and axis
plt.figure(figsize=(8,6))
plt.plot(tpr3,fpr3, label='zvv')
plt.plot(tpr2,fpr2, label='ztt')
plt.plot(tpr1,fpr1, label='vbfhh')
plt.plot(tpr4,fpr4, label='ttbar')
plt.legend()
plt.xlabel('met')
plt.ylabel('ratio')
plt.xlim(0,300)
plt.savefig('/abvol/eff_plots/met_efficiency2 (all mets, supervised).png')


# Create a new figure and axis
plt.figure(figsize=(8,6))
plt.plot(tpr3,fpr3, label='zvv')
plt.plot(tpr2,fpr2, label='ztt')
plt.plot(tpr1,fpr1, label='vbfhh')
plt.plot(tpr4,fpr4, label='ttbar')
plt.legend()
plt.xlabel('met')
plt.ylabel('ratio')
plt.yscale('log')
plt.ylim(0.0001,1)
plt.xlim(0,300)
plt.savefig('/abvol/eff_plots/met_efficiency2 (all mets, supervised - log).png')
