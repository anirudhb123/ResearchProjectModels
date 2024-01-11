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

print('xXx')

def transform(arr):
    a = len(arr)
    arr_reshaped = arr.reshape(a, 12, 2, 12, 2, 2)
    new_arr = np.sum(arr_reshaped, axis=(2, 4))

    return new_arr

print('xXx')


nom_bkg_jz0 = (transform(nom_bkg_jz0))[:,:,:,:1]
nom_bkg_jz1 = (transform(nom_bkg_jz1))[:,:,:,:1]
nom_bkg_jz2 = (transform(nom_bkg_jz2))[:,:,:,:1]


nom_sig_zvvhbb = (transform(nom_sig_zvvhbb))[:,:,:,:1]
nom_sig_ztautau = (transform(nom_sig_ztautau))[:,:,:,:1]

ajz0_met = np.load("/abvol/GepOutput_EM/jz0_pu200_met.npy", 'r')
ajz1_met = np.load("/abvol/GepOutput_EM/jz1_pu200_met.npy", 'r')
ajz2_met = np.load("/abvol/GepOutput_EM/jz2_pu200_met.npy", 'r')
azvvhbb_met = np.load("/abvol/GepOutput_EM/zvvhbb_pu200_met.npy", 'r')
aztautau_met = np.load("/abvol/GepOutput_EM/ztautau_pu200_met.npy", 'r')

print('xXx')


jz0_met = ajz0_met.copy()
jz1_met = ajz1_met.copy()
jz2_met = ajz2_met.copy()
zvvhbb_met = azvvhbb_met.copy()
ztautau_met = aztautau_met.copy()

print('xXx')


jz0_met[:,0] = jz0_met[:,0] / 1000
jz1_met[:,0] = jz1_met[:,0] / 1000
jz2_met[:,0] = jz2_met[:,0] / 1000
zvvhbb_met[:,0] = zvvhbb_met[:,0] / 1000
ztautau_met[:,0] = ztautau_met[:,0] / 1000



nom_data = np.concatenate([
    nom_bkg_jz0,
    nom_bkg_jz1])

nom_met = np.concatenate([jz0_met,jz1_met])

zvvhbb = nom_sig_zvvhbb.copy()
ztautau = nom_sig_ztautau.copy()

zvvhbb = nom_sig_zvvhbb
ztautau = nom_sig_ztautau

zvvhbb[:,:,:,0] = zvvhbb[:,:,:,0] / 1000
ztautau[:,:,:,0] = ztautau[:,:,:,0] / 1000

print('xXx')


# shuffle both datasets in same way
p = np.random.permutation(len(nom_data))
nom_data = nom_data[p]
nom_met = nom_met[p]

nom_data = nom_data[:args.nTotal]
nom_data[:,:,:,0] = nom_data[:,:,:,0]/1000. # scaling from mega to giga

nomTrainingDataLength = int(len(nom_data)*0.7)
nomValidationDataLength = int(len(nom_data)*0.1)
nomTestDataLength = int(len(nom_data)*0.2)

zvvhbbTrainingDataLength = int(len(zvvhbb)*0.7)
zvvhbbValidationDataLength = int(len(zvvhbb)*0.1)
zvvhbbTestDataLength = int(len(zvvhbb)*0.2)

print(len(nom_data))
print(len(nom_bkg_jz1))
print(len(nom_bkg_jz2))
print(len(zvvhbb))

modelName = "zvv_model_2"

input_cluster = Input(shape=(12, 12, 1), name='input_cluster')

x = Conv2D(8, 3, activation="relu", padding="same")(input_cluster)  # (12, 12, 8)
x = BatchNormalization()(x)
x = MaxPooling2D(2, padding="same")(x)  # (6, 6, 8)
x = BatchNormalization()(x)
x = MaxPooling2D(2, padding="same")(x)  # (3, 3, 8)
x = Flatten()(x)  # (576,)
x = Dense(24, activation="relu")(x)  # (32,)
x = Dense(12, activation="relu")(x)  # (16,)
x = Dense(4, activation="relu")(x)
x = Dense(2, activation="relu")(x)
output = Dense(1, activation="sigmoid", name="denseEndSix")(x)

nomClusterData = nom_data

trainingData = np.append(nomClusterData[0:nomTrainingDataLength],zvvhbb[0:zvvhbbTrainingDataLength],axis=0)

validationData = np.append(nomClusterData[nomTrainingDataLength:nomTrainingDataLength + nomValidationDataLength],zvvhbb[zvvhbbTrainingDataLength:zvvhbbTrainingDataLength + zvvhbbValidationDataLength],axis=0)

testData = np.append(nomClusterData[nomTrainingDataLength + nomValidationDataLength:nomTrainingDataLength + nomValidationDataLength + nomTestDataLength],zvvhbb[zvvhbbTrainingDataLength + zvvhbbValidationDataLength:zvvhbbTrainingDataLength + zvvhbbValidationDataLength + zvvhbbTestDataLength],axis=0)


trainingLabels = np.append(np.zeros(nomTrainingDataLength),np.ones(zvvhbbTrainingDataLength))                           
validationLabels = np.append(np.zeros(nomValidationDataLength),np.ones(zvvhbbValidationDataLength))
testLabels = np.append(np.zeros(nomTestDataLength),np.ones(zvvhbbTestDataLength))



##trainingData = np.append(nomClusterData[0:zvvhbbTrainingDataLength],zvvhbb[0:ztautauTrainingDataLength],axis=0)
##
##validationData = np.append(nomClusterData[zvvhbbTrainingDataLength:zvvhbbTrainingDataLength + zvvhbbValidationDataLength],zvvhbb[ztautauTrainingDataLength:ztautauTrainingDataLength + ztautauValidationDataLength],axis=0)
##
##testData = np.append(nomClusterData[zvvhbbTrainingDataLength + ztautauValidationDataLength:ztautauTrainingDataLength + zvvhbbValidationDataLength + ztautauTestDataLength],ztautau[ztautauTrainingDataLength + ztautauValidationDataLength:ztautauTrainingDataLength + ztautauValidationDataLength + ztautauTestDataLength],axis=0)
##
##
##trainingLabels = np.append(np.zeros(zvvhbbTrainingDataLength),np.ones(zvvhbbTrainingDataLength))                           
##validationLabels = np.append(np.zeros(zvvhbbValidationDataLength),np.ones(zvvhbbValidationDataLength))
##testLabels = np.append(np.zeros(zvvhbbTestDataLength),np.ones(zvvhbbTestDataLength))


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

#print("Compiling")

model = Model(inputs=input_cluster, outputs=output)
    
print(model.summary())

nom_met = nom_met[:,0]
zvvhbb_met = zvvhbb_met[:,0]

train_weights = np.power(np.reciprocal(np.concatenate((nom_met[0:nomTrainingDataLength],zvvhbb_met[0:zvvhbbTrainingDataLength]),axis=0)),3) * 1000
val_weights = np.power(np.reciprocal(np.concatenate((nom_met[nomTrainingDataLength:nomValidationDataLength],zvvhbb_met[zvvhbbTrainingDataLength:zvvhbbValidationDataLength]),axis=0)),3) * 1000


def replace_values(arr,threshold):
    mask = arr > threshold
    result = np.zeros_like(arr)    
    result[mask] = 0
    result[~mask] = 1
    return result

train_weights = replace_values(np.concatenate((nom_met[0:nomTrainingDataLength], zvvhbb_met[0:zvvhbbTrainingDataLength]), axis=0), 100)
val_weights = replace_values(np.concatenate((nom_met[nomTrainingDataLength:nomValidationDataLength], zvvhbb_met[zvvhbbTrainingDataLength:zvvhbbValidationDataLength]), axis=0), 100)


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[])
    
#print('Calculating')
    
modelCallbacks = [EarlyStopping(patience=25),
                    ModelCheckpoint(filepath="/abvol/Pred_zvv.hdf5", save_weights_only=True,
                                save_best_only=True)]


history = model.fit(x=trainingData, y=trainingLabels,epochs=numberOfEpochs, batch_size=batchSize,
                callbacks=modelCallbacks,
                validation_data=(validationData, validationLabels, val_weights),sample_weight=train_weights)
    
#print("Loading weights")
    
# model.load_weights("/abvol/Pred.hdf5")
    
model.save("/abvol/supervised_models/zvv"+modelName+",model")

for h in history.history:
    for ie in range(len(history.history[h])):
        history.history[h][ie] = float(history.history[h][ie])

#with open("/abvol/cluster_data/"+modelName+",history.json", "w") as f:
#    json.dump(history.history,f)



import matplotlib
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error

def weighted_loss(y_true, y_pred):
    y_pred = y_pred.flatten()
    epsilon = 1e-6  # A small value to prevent division by zero in log calculations
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # Clip to avoid invalid log values (0 or 1)
    return y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)


# predictions for each signal
preds = model.predict(testData)

for i in range(20):
    print(str(testLabels[i]) + ' / ' + str(preds[i,0]))


print(np.max(preds[:,0]))

num = 0

# for i in range(num):
#   preds = preds + model.predict(testData)

# nums = num + 1

# preds = np.array([i/nums for i in preds])


errors = weighted_loss(testLabels,np.array(preds))

errors = errors / np.sqrt(np.sum(errors**2)) # scaled

print('-----------------------------------------------------------------------------------------------')
print(np.array(preds).shape)
print(np.array(testLabels).shape)
print(errors.shape)
print('-----------------------------------------------------------------------------------------------')

from sklearn.metrics import roc_curve



fpr, tpr, thresholds = roc_curve(testLabels,preds)
auc = metrics.auc(fpr,tpr)

np.save('/abvol/fpr/fpr (zvv,supervised).npy',fpr)
np.save('/abvol/tpr/tpr (zvv,supervised).npy',tpr)

plt.figure(figsize=(8,6))

plt.plot(fpr,tpr,label='zvv, AUC = ' + str(auc))

 # FPR
plt.xscale(value='log')
plt.xlim(0.0001,1)

# TPR
plt.yscale(value='log')
plt.ylim(0.001,1)

plt.legend()
plt.savefig('/abvol/ROC_Curves/ROC Curve (zvv,supervised).png')

print(auc)

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


threshold_idx1 = find_nearest(fpr,value=0.0001)
threshold1 = thresholds[threshold_idx1]

print(threshold1)

threshold_idx2 = find_nearest(fpr,value=0.001)
threshold2 = thresholds[threshold_idx2]
print(threshold2)

#----------------------------------------------------------------------------------------

indices1 = np.where(preds > threshold1)[0]

print(str(len(zvvhbb_met) * threshold1) + " / " + str(len(zvvhbb_met)))

indices2 = np.where(preds > threshold2)[0]

print(str(len(zvvhbb_met) * threshold2) + " / " + str(len(zvvhbb_met)))

mets = np.concatenate([nom_met[(-1 * nomTestDataLength):], zvvhbb_met[(-1 * zvvhbbTestDataLength):]])

valid_mets = np.array(mets[indices1])

plt.figure(figsize=(8,6))
bin_vals_1, bins_1, _ = plt.hist(mets, bins=200, histtype='step', label='all events')
bin_vals_2, _, _ = plt.hist(valid_mets, bins=bins_1, histtype='step', label='events w/ disc > 10^-4')
plt.xlabel('met')
plt.ylabel('frequency')
plt.legend()
plt.savefig('/abvol/eff_plots/met_efficiency (zvv, supervised).png')

plt.figure(figsize=(8,6))
bin_vals_1,bins_1,_ = plt.hist(mets,bins=200,histtype='step',label='all events')
bin_vals_2,_,_ = plt.hist(valid_mets,bins=bins_1,histtype='step',label='events w/ disc > 10^-4')
plt.xlabel('met')
plt.ylabel('frequency')
plt.yscale('log')
plt.xlim(0,300)
plt.legend()
plt.savefig('/abvol/eff_plots/met_efficiency (zvv, supervised), log.png')


divisors = []

for i in range(bin_vals_1.shape[0]):
    if bin_vals_1[i] != 0:
        divisors.append(bin_vals_2[i] / bin_vals_1[i])
    else:
        divisors.append(0)

divisors = np.array(divisors)
bins1 = []
for i in range(1,len(bins_1)):
    bins1.append(int((bins_1[i] + bins_1[i-1]) / 2))

bins1 = np.array(bins1)
divisors = np.array(divisors)

np.save('/abvol/bins/(zvv,supervised).npy', bins1)
np.save('/abvol/divisors/(zvv,supervised).npy', divisors)

plt.figure(figsize=(8,6))
plt.plot(bins1,divisors)
plt.legend()
plt.xlabel('met')
plt.ylabel('ratio')
plt.xlim(0,300)
plt.ylim(0,1)
plt.savefig('/abvol/eff_plots/met_efficiency2 (zvv, supervised).png')

plt.figure(figsize=(8,6))
plt.yscale('log')
plt.plot(bins1,divisors)
plt.legend()
plt.xlabel('met')
plt.ylabel('ratio')
plt.xlim(0,300)
plt.ylim(0.01,1)
plt.savefig('/abvol/eff_plots/met_efficiency2 (zvv, supervised), log.png')
