# Imports basics
import numpy as np
import pandas as pd
import h5py
import keras.backend as K
import tensorflow as tf
import json
import random

from sklearn import metrics

# Imports neural net tools

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input, Dense, GRU, Add, Concatenate, BatchNormalization, Conv2D, Lambda, Dot, Flatten, Layer, Reshape, MaxPooling2D,Dropout
from keras.models import Model, load_model

# Opens files and reads data

clustersTotal = 200
entriesPerCluster = 3 # pt, eta, phi

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--label', action="store", dest="label", type=str, default="")
parser.add_argument('--nTrain', action="store", dest="nTrain", type=int, default=-1)
parser.add_argument('--nTotal', action="store", dest="nTotal", type=int, default=-1)
parser.add_argument('--nEpochs', action="store", dest="nEpochs", type=int, default=200) # change later but do 50 for now
parser.add_argument('--clusters', action="store", dest="clusters", type=int, default=100)
parser.add_argument('--batchSize', action="store", dest="batchSize", type=int, default=256)
parser.add_argument('--doAttention', action="store_true", dest="doAttention")
parser.add_argument('--useSig', action="store_true", dest="useSig")
parser.add_argument('--onlyPlot', action="store_true", dest="onlyPlot")

args = parser.parse_args()

clustersConsidered = args.clusters

numberOfEpochs = args.nEpochs
batchSize = args.batchSize


np.random.seed(422022)

nom_bkg_jz0 = np.load("/abvol/GepOutput_EM/jz0_pu200_grid.npy", 'r')
nom_bkg_jz1 = np.load("/abvol/GepOutput_EM/jz1_pu200_grid.npy", 'r')
nom_bkg_jz2 = np.load("/abvol/GepOutput_EM/jz2_pu200_grid.npy", 'r')
nom_sig_zvvhbb = np.load("/abvol/GepOutput_EM/zvvhbb_pu200_grid.npy", 'r')
nom_sig_ztautau = np.load("/abvol/GepOutput_EM/ztautau_pu200_grid.npy", 'r')
nom_sig_vbfhh = np.load("/abvol/GepOutput_EM/vbfhhbbbb_pu200_grid.npy", 'r')
nom_sig_ttbar = np.load("/abvol/GepOutput_EM/ttbar_pu200_grid.npy", 'r')

print(nom_sig_ttbar.shape)
print(nom_sig_vbfhh.shape)

def transform(arr):
    a = len(arr)
    arr_reshaped = arr.reshape(a, 12, 2, 12, 2, 2)
    new_arr = np.sum(arr_reshaped, axis=(2, 4))
    return new_arr


nom_bkg_jz0 = (transform(nom_bkg_jz0))[:,:,:,:1]
nom_bkg_jz1 = (transform(nom_bkg_jz1))[:,:,:,:1]
nom_bkg_jz2 = (transform(nom_bkg_jz2))[:,:,:,:1]


nom_sig_zvvhbb = (transform(nom_sig_zvvhbb))[:,:,:,:1]
nom_sig_ztautau = (transform(nom_sig_ztautau))[:,:,:,:1]
nom_sig_vbfhh = (transform(nom_sig_vbfhh))[:,:,:,:1]
nom_sig_ttbar = (transform(nom_sig_ttbar))[:,:,:,:1]

ajz0_met = np.load("/abvol/GepOutput_EM/jz0_pu200_met.npy", 'r')
ajz1_met = np.load("/abvol/GepOutput_EM/jz1_pu200_met.npy", 'r')
ajz2_met = np.load("/abvol/GepOutput_EM/jz2_pu200_met.npy", 'r')
azvvhbb_met = np.load("/abvol/GepOutput_EM/zvvhbb_pu200_met.npy", 'r')
aztautau_met = np.load("/abvol/GepOutput_EM/ztautau_pu200_met.npy", 'r')
avbfhh_met = np.load("/abvol/GepOutput_EM/vbfhhbbbb_pu200_met.npy", 'r')
attbar_met = np.load("/abvol/GepOutput_EM/ttbar_pu200_met.npy", 'r')


jz0_met = ajz0_met.copy()
jz1_met = ajz1_met.copy()
jz2_met = ajz2_met.copy()
zvvhbb_met = azvvhbb_met.copy()
ztautau_met = aztautau_met.copy()
vbfhh_met = avbfhh_met.copy()
ttbar_met = attbar_met.copy()

jz0_met[:,0] = jz0_met[:,0] / 1000
jz1_met[:,0] = jz1_met[:,0] / 1000
jz2_met[:,0] = jz2_met[:,0] / 1000
zvvhbb_met[:,0] = zvvhbb_met[:,0] / 1000
ztautau_met[:,0] = ztautau_met[:,0] / 1000
vbfhh_met[:,0] = vbfhh_met[:,0] / 1000
ttbar_met[:,0] = ttbar_met[:,0] / 1000


nom_data = np.concatenate([
nom_bkg_jz0,
nom_bkg_jz1])

nom_met = np.concatenate([jz0_met,jz1_met])

zvvhbb = nom_sig_zvvhbb.copy()
ztautau = nom_sig_ztautau.copy()
vbfhh = nom_sig_vbfhh.copy()
ttbar = nom_sig_ttbar.copy()

zvvhbb[:,:,:,0] = zvvhbb[:,:,:,0] / 1000
ztautau[:,:,:,0] = ztautau[:,:,:,0] / 1000
vbfhh[:,:,:,0] = vbfhh[:,:,:,0] / 1000
ttbar[:,:,:,0] = ttbar[:,:,:,0] / 1000


nom_data = np.concatenate([
nom_bkg_jz0,
nom_bkg_jz1])

nom_met = np.concatenate([jz0_met,jz1_met])

# shuffle both datasets in same way
p = np.random.permutation(len(nom_data))
nom_data = nom_data[p]
nom_met = nom_met[p]


nom_met = nom_met[:,0]
zvvhbb_met = zvvhbb_met[:,0]
ztautau_met = ztautau_met[:,0]
ttbar_met = ttbar_met[:,0]
vbfhh_met = vbfhh_met[:,0]

signal_mets = np.concatenate([zvvhbb_met,ztautau_met,ttbar_met,vbfhh_met])


nom_data = nom_data[:args.nTotal]
nom_data[:,:,:,0] = nom_data[:,:,:,0]/1000. # scaling from mega to giga

nomTrainingDataLength = int(len(nom_data)*0.7)
nomValidationDataLength = int(len(nom_data)*0.1)
nomTestDataLength = int(len(nom_data)*0.2)

signalTrainingDataLength = int(len(signal_mets)*0.7)
signalValidationDataLength = int(len(signal_mets)*0.1)
signalTestDataLength = int(len(signal_mets)*0.2)


print(len(nom_data))
print(len(nom_bkg_jz1))
print(len(nom_bkg_jz2))
print(len(vbfhh))

modelName = "vbfhh_met_model_1"

input_cluster = Input(shape=(1), name='input_cluster')


nomClusterData = nom_data

window_l = 0
window_h = 1000


nom_t = nom_met[0:nomTrainingDataLength]

#nom_t = nom_t[nom_t <= window_h]

signal_t = signal_mets[0:signalTrainingDataLength]

#signal_t = signal_t[signal_t >= window_l]


nom_v = nom_met[nomTrainingDataLength:nomTrainingDataLength + nomValidationDataLength]

#nom_v = nom_v[nom_v <= window_h]

signal_v = signal_mets[signalTrainingDataLength:signalTrainingDataLength + signalValidationDataLength]

#signal_v = signal_v[signal_v >= window_l]

nom_te = nom_met[nomTrainingDataLength + nomValidationDataLength:nomTrainingDataLength + nomValidationDataLength + nomTestDataLength]

#nom_te = nom_te[nom_te <= window_h]

signal_te = signal_mets[signalTrainingDataLength + signalValidationDataLength:signalTrainingDataLength + signalValidationDataLength + signalTestDataLength]

#signal_te = signal_te[signal_te >= window_l]


trainingData = np.append(nom_t, signal_t, axis=0)

validationData = np.append(nom_v, signal_v, axis=0)

testData = np.append(nom_te, signal_te, axis=0)

trainingData = trainingData / np.sum(trainingData)
validationData = validationData / np.sum(validationData)
testData = testData / np.sum(testData)

trainingLabels = np.append(np.zeros(len(nom_t)), np.ones(len(signal_t)))
validationLabels = np.append(np.zeros(len(nom_v)), np.ones(len(signal_v)))
testLabels = np.append(np.zeros(len(nom_te)), np.ones(len(signal_te)))

p2 = np.random.permutation(len(trainingData))
trainingData = trainingData[p2]
trainingLabels = trainingLabels[p2]

p3 = np.random.permutation(len(validationData))
validationData = validationData[p3]
validationLabels = validationLabels[p3]

p4 = np.random.permutation(len(testData))
testData = testData[p4]
testLabels = testLabels[p4]

print(trainingLabels.shape)

model = Model(inputs=input_cluster, outputs=input_cluster)

print(model.summary())

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[])



modelCallbacks = [EarlyStopping(patience=25),
              ModelCheckpoint(filepath="/abvol/Pred_mets.hdf5", save_weights_only=True,
                              save_best_only=True)]

history = model.fit(x=trainingData, y=trainingLabels, epochs=numberOfEpochs, batch_size=batchSize,
                callbacks=modelCallbacks,
                validation_data=(validationData, validationLabels))

model.save("/abvol/supervised_models/mets" + modelName + ",model")

for h in history.history:
    for ie in range(len(history.history[h])):
        history.history[h][ie] = float(history.history[h][ie])

import matplotlib
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error

def weighted_loss(y_true, y_pred):
    y_pred = y_pred.flatten()
    epsilon = 1e-6
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)

preds = model.predict(testData)

for i in range(20):
    print(str(testLabels[i]) + ' / ' + str(preds[i,0]))


num = 0

errors = weighted_loss(testLabels, np.array(preds))

errors = errors / np.sqrt(np.sum(errors**2))

print('-----------------------------------------------------------------------------------------------')
print(np.array(preds).shape)
print(np.array(testLabels).shape)
print(errors.shape)
print('-----------------------------------------------------------------------------------------------')

from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(testLabels, preds)
auc = metrics.auc(fpr, tpr)

np.save('/abvol/fpr/fpr (met,supervised).npy', fpr)
np.save('/abvol/tpr/tpr (met,supervised).npy', tpr)

plt.figure(figsize=(8,6))

plt.plot(fpr, tpr, label='met, AUC = ' + str(auc))

plt.xscale(value='log')
plt.xlim(0.0001, 1)

plt.yscale(value='log')
plt.ylim(0.001, 1)

plt.legend()
plt.savefig('/abvol/ROC_Curves/ROC Curve (met,supervised).png')


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

t_val = 0.01
threshold_idx1 = find_nearest(fpr, value=t_val)
threshold1 = thresholds[threshold_idx1]


threshold_idx2 = find_nearest(fpr, value=0.001)
threshold2 = tpr[threshold_idx2]

#-----------------------------------------------

indices1 = np.where(preds > threshold1)[0]

indices2 = np.where(preds > threshold2)[0]


mets = testData

valid_mets = np.array(testData[indices1])


print(len(valid_mets))
print(len(mets))

np.save('/abvol/mets/mets (met,supervised).npy', mets)
np.save('/abvol/val_mets/val_mets (met,supervised).npy', valid_mets)

plt.figure(figsize=(8,6))
bin_vals_1, bins_1, _ = plt.hist(mets, bins=200, histtype='step', label='all events')
bin_vals_2, _, _ = plt.hist(valid_mets, bins=bins_1, histtype='step', label='events w/ disc > 10^-4')
plt.xlabel('met')
plt.ylabel('frequency')
plt.legend()
plt.savefig('/abvol/eff_plots/met_efficiency (met,supervised).png')

plt.figure(figsize=(8,6))
bin_vals_1,bins_1,_ = plt.hist(mets,bins=200,histtype='step',label='all events')
bin_vals_2,_,_ = plt.hist(valid_mets,bins=bins_1,histtype='step',label='events w/ disc > 10^-4')
plt.xlabel('met')
plt.ylabel('frequency')
plt.yscale('log')
plt.xlim(0,300)
plt.legend()
plt.savefig('/abvol/eff_plots/met_efficiency (met,supervised), log.png')


divisors = []

for i in range(bin_vals_1.shape[0]):
    if bin_vals_1[i] != 0:
        divisors.append(bin_vals_2[i] / bin_vals_1[i])
    else:
        divisors.append(0)

bins1 = []
for i in range(1,len(bins_1)):
    bins1.append(int((bins_1[i] + bins_1[i-1]) / 2))

bins1 = np.array(bins1)
divisors = np.array(divisors)


np.save('/abvol/bins/(met,supervised).npy', bins1)
np.save('/abvol/divisors/(met,supervised).npy', divisors)

plt.figure(figsize=(8,6))
plt.plot(bins1,divisors)
plt.legend()
plt.xlabel('met')
plt.ylabel('ratio')
plt.xlim(0,300)
plt.ylim(0,1)
plt.savefig('/abvol/eff_plots/met_efficiency2 (met,supervised).png')

plt.figure(figsize=(8,6))
plt.yscale('log')
plt.plot(bins1,divisors)
plt.legend()
plt.xlabel('met')
plt.ylabel('ratio')
plt.xlim(0,300)
plt.ylim(0.01,1)
plt.savefig('/abvol/eff_plots/met_efficiency2 (met,supervised), log.png')
