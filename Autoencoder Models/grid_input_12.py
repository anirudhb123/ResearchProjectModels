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
from keras.layers import Input, Dense, GRU, Add, Concatenate, BatchNormalization, Conv2D, Conv2DTranspose, Lambda, Dot, Flatten, Layer, Reshape, MaxPooling1D,Dropout,MaxPooling2D,Conv2D,UpSampling2D,ReLU,LeakyReLU
from keras.models import Model, load_model

from tensorflow.keras import regularizers



from sklearn.metrics import mean_squared_error
from scipy.stats import wasserstein_distance

##from comet_ml import Experiment
##experiment = Experiment(
##  api_key = "7esN7aVWrTUFyYSzrypbF9bWX",
##  project_name = "general",
##  workspace="abharadwaj123"
##)
##
##loss_func = 'mse'
##
##params= {
##    'batch_size':64,
##    'epochs':100,
##    'latent_space_size':12,
##    'loss_function':loss_func
##}


# Opens files and reads data

clustersTotal = 200
entriesPerCluster = 3 # pt, eta, phi

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--label', action="store", dest="label", type=str, default="")
parser.add_argument('--nTrain', action="store", dest="nTrain", type=int, default=-1)
parser.add_argument('--nTotal', action="store", dest="nTotal", type=int, default=-1)
parser.add_argument('--nEpochs', action="store", dest="nEpochs", type=int, default=200) # change later but do 50 for now
parser.add_argument('--clusters', action="store", dest="clusters", type=int, default=200)
parser.add_argument('--batchSize', action="store", dest="batchSize", type=int, default=512)
parser.add_argument('--doAttention', action="store_true", dest="doAttention")
parser.add_argument('--useSig', action="store_true", dest="useSig")
parser.add_argument('--onlyPlot', action="store_true", dest="onlyPlot")

args = parser.parse_args()

clustersConsidered = args.clusters

numberOfEpochs = args.nEpochs
batchSize = args.batchSize

#print(args)

np.random.seed(422022)

# print("Extracting")

nom_bkg_jz0 = np.load("/abvol/GepOutput/jz0_pu200_clusters.npy", 'r')
nom_bkg_jz1 = np.load("/abvol/GepOutput/jz1_pu200_clusters.npy", 'r')
nom_bkg_jz2 = np.load("/abvol/GepOutput/jz2_pu200_clusters.npy", 'r')
nom_sig_zvvhbb = np.load("/abvol/GepOutput/zvvhbb_pu200_clusters.npy", 'r')
nom_sig_ztautau = np.load("/abvol/GepOutput/ztautau_pu200_clusters.npy", 'r')
nom_sig_zprimetthad = np.load("/abvol/GepOutput/zprimetthad_pu200_clusters.npy", 'r')
nom_sig_vbfhh = np.load("/abvol/GepOutput_EM/vbfhhbbbb_pu200_grid.npy", 'r')
nom_sig_ttbar = np.load("/abvol/GepOutput_EM/ttbar_pu200_grid.npy", 'r')

ajz0_met = np.load("/abvol/GepOutput/jz0_pu200_met.npy", 'r')
ajz1_met = np.load("/abvol/GepOutput/jz1_pu200_met.npy", 'r')
ajz2_met = np.load("/abvol/GepOutput/jz2_pu200_met.npy", 'r')
azvvhbb_met = np.load("/abvol/GepOutput/zvvhbb_pu200_met.npy", 'r')
aztautau_met = np.load("/abvol/GepOutput/ztautau_pu200_met.npy", 'r')
azprimetthad_met = np.load("/abvol/GepOutput/zprimetthad_pu200_met.npy", 'r')
avbfhh_met = np.load("/abvol/GepOutput_EM/vbfhhbbbb_pu200_met.npy", 'r')
attbar_met = np.load("/abvol/GepOutput_EM/ttbar_pu200_met.npy", 'r')

jz0_met = ajz0_met.copy()
jz1_met = ajz1_met.copy()
jz2_met = ajz2_met.copy()
zvvhbb_met = azvvhbb_met.copy()
ztautau_met = aztautau_met.copy()
zprimetthad_met = azprimetthad_met.copy()
vbfhh_met = avbfhh_met.copy()
ttbar_met = attbar_met.copy()

jz0_met[:,0] = jz0_met[:,0] / 1000
jz1_met[:,0] = jz1_met[:,0] / 1000
jz2_met[:,0] = jz2_met[:,0] / 1000
zvvhbb_met[:,0] = zvvhbb_met[:,0] / 1000
ztautau_met[:,0] = ztautau_met[:,0] / 1000
zprimetthad_met[:,0] = zprimetthad_met[:,0] / 1000
vbfhh_met[:,0] = vbfhh_met[:,0] / 1000
ttbar_met[:,0] = ttbar_met[:,0] / 1000

#nom_data = np.concatenate([
#    nom_bkg_jz0,
#    nom_bkg_jz1,
#    nom_bkg_jz2 ])

nom_data = np.concatenate([
    nom_bkg_jz0,
    nom_bkg_jz1])

nom_met = np.concatenate([jz0_met,jz1_met])

zvvhbb = nom_sig_zvvhbb.copy()
ztautau = nom_sig_ztautau.copy()
zprimetthad = nom_sig_zprimetthad.copy()
vbfhh = nom_sig_vbfhh.copy()
ttbar = nom_sig_ttbar.copy()

zvvhbb[:,:,0] = zvvhbb[:,:,0] / 1000
ztautau[:,:,0] = ztautau[:,:,0] / 1000
zprimetthad[:,:,0] = zprimetthad[:,:,0] / 1000
vbfhh[:,:,:,0] = vbfhh[:,:,:,0] / 1000
ttbar[:,:,:,0] = ttbar[:,:,:,0] / 1000


# shuffle both datasets in same way
p = np.random.permutation(len(nom_data))
nom_data = nom_data[p]
nom_met = nom_met[p]

nom_data = nom_data[:args.nTotal]
nom_data[:,:,0] = nom_data[:,:,0]/1000. # scaling from mega to giga

for x in nom_data:
    x = x[x[:,0].argsort()]
    # x = np.flip(x, axis=0)

sig = np.concatenate([
    zvvhbb,
    zprimetthad,
    ztautau
    ])

sig_met = np.concatenate([zvvhbb_met,zprimetthad_met,ztautau_met])

for x in sig:
    x = x[-x[:,0].argsort()]
    # x = np.flip(x, axis=0)

nomTrainingDataLength = int(len(nom_data)*0.7)
nomValidationDataLength = int(len(nom_data)*0.1)
nomTestingDataLength = len(nom_data) - (nomTrainingDataLength + nomValidationDataLength)


nomClusterData = nom_data
training = nomClusterData[0:nomTrainingDataLength]
validation = nomClusterData[nomTrainingDataLength:nomTrainingDataLength + nomValidationDataLength]
testing = nomClusterData[-nomTestingDataLength:]

nomClusterTrainingData = []
nomClusterValidationData = []
nomClusterTestData = []

nomMetTraining = nom_met[0:nomTrainingDataLength]
nomMetValidation = nom_met[nomTrainingDataLength:nomTrainingDataLength+nomValidationDataLength]
nomMetTesting = nom_met[-nomTestingDataLength:]

eta_bins = np.linspace(-3,3,num=13) # 12
phi_bins = np.linspace(-np.pi,np.pi,num=13) # 12

for i in range(training.shape[0]):
    a,_,_ = np.histogram2d(training[i,:,1].flatten(),training[i,:,2].flatten(), bins=[eta_bins,phi_bins],weights=training[i,:,0].flatten())

    avg_met = nomMetTraining[i,0] / 144
    phi = nomMetTraining[i,1]

    a = np.expand_dims(a,axis=2)
    a = np.insert(a,1,avg_met,axis=2)
    a = np.insert(a,2,phi,axis=2)
   
    nomClusterTrainingData.append(a)

nomClusterTrainingData = np.array(nomClusterTrainingData)

def augment_data(data, num_augmentations):
    augmented_data = []
    for sample in data:
        augmented_data.append(sample)
        for _ in range(num_augmentations):
            augmented_sample = sample + np.random.normal(loc=0, scale=0.01, size=sample.shape)
            augmented_data.append(augmented_sample)
    return np.array(augmented_data)

num_augmentations = 2
nomClusterTrainingData = augment_data(nomClusterTrainingData, num_augmentations)


for j in range(validation.shape[0]):
    a,_,_ = np.histogram2d(validation[j,:,1].flatten(),validation[j,:,2].flatten(), bins=[eta_bins,phi_bins],weights=validation[j,:,0].flatten())
    
    avg_met = nomMetValidation[j,0] / 144
    phi = nomMetValidation[j,1]

    a = np.expand_dims(a,axis=2)
    a = np.insert(a,1,avg_met,axis=2)
    a = np.insert(a,2,phi,axis=2)
                  
    nomClusterValidationData.append(a)

nomClusterValidationData = np.array(nomClusterValidationData)    


for k in range(testing.shape[0]):
    a,_,_ = np.histogram2d(testing[k,:,1].flatten(),testing[k,:,2].flatten(), bins=[eta_bins,phi_bins],weights=testing[k,:,0].flatten())

    avg_met = nomMetTesting[k,0] / 144
    phi = nomMetTesting[k,1]

    a = np.expand_dims(a,axis=2)
    a = np.insert(a,1,avg_met,axis=2)
    a = np.insert(a,2,phi,axis=2)
    
    nomClusterTestData.append(a)

nomClusterTestData = np.array(nomClusterTestData)    


#nomClusterTrainingData = np.expand_dims(np.array(nomClusterTrainingData),axis=(3))
#nomClusterValidationData = np.expand_dims(np.array(nomClusterValidationData),axis=(3))
#nomClusterTestData = np.expand_dims(np.array(nomClusterTestData),axis=(3))

nomClusterTrainingLabels = nomClusterTrainingData.copy()
nomClusterValidationLabels = nomClusterValidationData.copy()
nomClusterTestLabels = nomClusterTestData.copy()


nom_sig = []

for l in range(sig.shape[0]):
    a,_,_ = np.histogram2d(sig[l,:,1].flatten(),sig[l,:,2].flatten(), bins=[eta_bins,phi_bins],weights=sig[l,:,0].flatten())

    avg_met = sig_met[l,0] / 144
    phi = sig_met[l,1]

    a = np.expand_dims(a,axis=2)
    a = np.insert(a,1,avg_met,axis=2)
    a = np.insert(a,2,phi,axis=2)
    
    nom_sig.append(a)

nom_sig = np.array(nom_sig)
print(nom_sig.shape)


sig_zvvhbb = []
sig_ztautau = []
sig_zprimetthad = []
sig_vbfhh = []
sig_ttbar = []

for i in range(zvvhbb.shape[0]):
    a,_,_ = np.histogram2d(zvvhbb[i,:,1].flatten(),zvvhbb[i,:,2].flatten(), bins=[eta_bins,phi_bins],weights=zvvhbb[i,:,0].flatten())

    avg_met = zvvhbb_met[i,0] / 144
    phi = zvvhbb_met[i,1]

    a = np.expand_dims(a,axis=2)
    a = np.insert(a,1,avg_met,axis=2)
    a = np.insert(a,2,phi,axis=2)
    
    sig_zvvhbb.append(a)

for j in range(ztautau.shape[0]):
    a,_,_ = np.histogram2d(ztautau[j,:,1].flatten(),ztautau[j,:,2].flatten(), bins=[eta_bins,phi_bins],weights=ztautau[j,:,0].flatten())

    avg_met = ztautau_met[j,0] / 144
    phi = ztautau_met[j,1]

    a = np.expand_dims(a,axis=2)
    a = np.insert(a,1,avg_met,axis=2)
    a = np.insert(a,2,phi,axis=2)
    
    sig_ztautau.append(a)

for k in range(zprimetthad.shape[0]):
    a,_,_ = np.histogram2d(zprimetthad[k,:,1].flatten(),zprimetthad[k,:,2].flatten(), bins=[eta_bins,phi_bins],weights=zprimetthad[k,:,0].flatten())

    avg_met = zprimetthad_met[k,0] / 144
    phi = zprimetthad_met[k,1]

    a = np.expand_dims(a,axis=2)
    a = np.insert(a,1,avg_met,axis=2)
    a = np.insert(a,2,phi,axis=2)
    
    sig_zprimetthad.append(a)

for i in range(vbfhh.shape[0]):
    a, _, _ = np.histogram2d(vbfhh[i, :, 1].flatten(), vbfhh[i, :, 2].flatten(), bins=[eta_bins, phi_bins], weights=vbfhh[i, :, 0].flatten())

    avg_met = vbfhh_met[i, 0] / 144
    phi = vbfhh_met[i, 1]

    a = np.expand_dims(a, axis=2)
    a = np.insert(a, 1, avg_met, axis=2)
    a = np.insert(a, 2, phi, axis=2)

    sig_vbfhh.append(a)


for i in range(ttbar.shape[0]):
    a, _, _ = np.histogram2d(ttbar[i, :, 1].flatten(), ttbar[i, :, 2].flatten(), bins=[eta_bins, phi_bins], weights=ttbar[i, :, 0].flatten())

    avg_met = ttbar_met[i, 0] / 144
    phi = ttbar_met[i, 1]

    a = np.expand_dims(a, axis=2)
    a = np.insert(a, 1, avg_met, axis=2)
    a = np.insert(a, 2, phi, axis=2)

    sig_ttbar.append(a)

sig_zvvhbb = np.array(sig_zvvhbb)
sig_ztautau = np.array(sig_ztautau)
sig_zprimetthad = np.array(sig_zprimetthad)
sig_vbfhh = np.array(sig_vbfhh)
sig_ttbar = np.array(sig_ttbar)

#sig_zvvhbb = np.expand_dims(np.array(sig_zvvhbb),axis=(3))
#sig_ztautau = np.expand_dims(np.array(sig_ztautau),axis=(3))
#sig_zprimetthad = np.expand_dims(np.array(sig_zprimetthad),axis=(3))

from tensorflow.keras.regularizers import l2

modelName = "Pred"

# Input
inputCluster = Input(shape=(12, 12, 3), name='inputCluster')

# Encoder
conv1 = Conv2D(32, (3, 3), padding='same')(inputCluster)
conv1 = BatchNormalization()(conv1)
conv1 = LeakyReLU()(conv1)
pool1 = MaxPooling2D((2, 2))(conv1) #(6,6,32)

conv2 = Conv2D(64, (3, 3), padding='same')(pool1)
conv2 = BatchNormalization()(conv2)
conv2 = LeakyReLU()(conv2)
pool2 = MaxPooling2D((2, 2))(conv2) #(3,3,64)

# Latent Space
flatten = Flatten()(pool2)
dense1 = Dense(312)(flatten)
dense1 = LeakyReLU()(dense1)
dense2 = Dense(576)(dense1)
dense2 = ReLU()(dense2)

# Decoder
reshape = Reshape((3, 3, 64))(dense2)

conv_trans1 = Conv2DTranspose(64, (3, 3), padding='same')(reshape)
conv_trans1 = BatchNormalization()(conv_trans1)
conv_trans1 = LeakyReLU()(conv_trans1)
upsample1 = UpSampling2D((2, 2))(conv_trans1)

conv_trans2 = Conv2DTranspose(32, (3, 3), padding='same')(upsample1)
conv_trans2 = BatchNormalization()(conv_trans2)
conv_trans2 = LeakyReLU()(conv_trans2)
upsample2 = UpSampling2D((2, 2))(conv_trans2) #(12,12,32)

conv_trans3 = Conv2DTranspose(3, (3, 3), padding='same')(upsample2)

output = conv_trans3

#print("Preparing Data")

clusterDataLength = clustersTotal


#print("Compiling")

model = Model(inputs=inputCluster, outputs=output)
    
print(model.summary())

print(nomClusterTrainingData.shape)
print(nomClusterValidationData.shape)
print(nomClusterTestData.shape)

print(nomMetTraining.shape)
print(nomMetValidation.shape)
print(nomMetTesting.shape)

fpr_weight = 0.1

def custom_loss(y_true, y_pred):
    weight_factor = fpr_weight
    mse = K.square(y_true - y_pred)    
    weighted_mse = tf.where(K.less(y_true, y_pred), weight_factor * mse, mse)
    return K.mean(weighted_mse, axis=[1, 2, 3])

model.compile(optimizer='adam', loss=custom_loss)
    
#print('Calculating')
    
modelCallbacks = [EarlyStopping(min_delta=0.001,patience=25),
                    ModelCheckpoint(filepath="/abvol/Pred.hdf5", save_weights_only=True,
                                save_best_only=True)]

#with experiment.train():
history = model.fit(x=nomClusterTrainingData, y=nomClusterTrainingLabels,epochs=numberOfEpochs, batch_size=batchSize,
                callbacks=modelCallbacks,
                validation_data=(nomClusterValidationData, nomClusterValidationLabels))
    
#print("Loading weights")
    
model.load_weights("/abvol/Pred.hdf5")
    
#model.save("/abvol/cluster_models/"+modelName+",model")

for h in history.history:
    for ie in range(len(history.history[h])):
        history.history[h][ie] = float(history.history[h][ie])

#with open("/abvol/cluster_data/"+modelName+",history.json", "w") as f:
#    json.dump(history.history,f)

##def mean_vals(arrays):
##    return np.mean(arrays, axis=0)
##
##dijet_preds = []
##for i in range(5):
##    dijet_preds.append(model.predict(nomClusterTestData))
##
##dijet_preds = np.array(mean_vals(dijet_preds))
##print(dijet_preds.shape)

#sig_preds = model.predict(nom_sig)

#rmse_dijets = mean_squared_error(dijet_preds,nomClusterTestData)
#rmse_sig = mean_squared_error(sig_preds,nom_sig)

#rmse_dijets = np.mean((np.sum((dijet_preds - nomClusterTestData)**2)))
#rmse_sig = np.mean((np.sum((sig_preds - nom_sig)**2)))

#print("MSE ON DIJETS: " + str(rmse_dijets))
#print("MSE ON SIGNATURE MODELS: " + str(rmse_sig))


import matplotlib
import matplotlib.pyplot as plt

# predictions for each signal

#sig_zvvhbb_preds = []
#for i in range(5):
#    sig_zvvhbb_preds.append(model.predict(sig_zvvhbb))
#sig_zvvhbb_preds = np.array(mean_vals(sig_zvvhbb_preds))

#sig_zprimetthad_preds = []
#for i in range(5):
#    sig_zprimetthad_preds.append(model.predict(sig_zprimetthad))
#sig_zprimetthad_preds = np.array(mean_vals(sig_zprimetthad_preds))

#sig_ztautau_preds = []
#for i in range(5):
#    sig_ztautau_preds.append(model.predict(sig_ztautau))
#sig_ztautau_preds = np.array(mean_vals(sig_ztautau_preds))

#print(sig_zvvhbb_preds.shape)
#print(sig_zprimetthad_preds.shape)
#print(sig_ztautau_preds.shape)

def weighted_loss(y_true, y_pred, weight_factor=fpr_weight):
    mse = np.square(y_true - y_pred)
    weighted_mse = np.where(y_pred > y_true, weight_factor * mse, mse)
    return np.mean(weighted_mse, axis=(1, 2, 3))

errors_nomTestData = weighted_loss(nomClusterTestData,model.predict(nomClusterTestData))
errors_sig_zvvhbb = weighted_loss(sig_zvvhbb,model.predict(sig_zvvhbb))
errors_sig_zprimetthad = weighted_loss(sig_zprimetthad,model.predict(sig_zprimetthad))
errors_sig_ztautau = weighted_loss(sig_ztautau,model.predict(sig_ztautau))
errors_sig_vbfhh = weighted_loss(sig_vbfhh,model.predict(sig_vbfhh))
errors_sig_ttbar = weighted_loss(sig_ttbar,model.predict(sig_ttbar))

##
##for i in range(nomClusterTestData.shape[0]):
##    a = nomClusterTestData[i]
##    
##    a = a.flatten()
##    d = dijet_preds[i]
##    d = d.flatten()                                                                              
##    errors_nomTestData = np.append(errors_nomTestData, mean_squared_error(a,d))
##
##
##for i in range(sig_zvvhbb.shape[0]):
##    g = sig_zvvhbb[i]
##    g = g.flatten()
##    k = sig_zvvhbb_preds[i]
##    k = k.flatten()                                                                
##    errors_sig_zvvhbb = np.append(errors_sig_zvvhbb, mean_squared_error(g,k))
##
##
##for i in range(sig_zprimetthad.shape[0]):
##    n = sig_zprimetthad[i]
##    n = n.flatten()
##    q = sig_zprimetthad[i]
##    q = q.flatten()                                                        
##    errors_sig_zprimetthad = np.append(errors_sig_zprimetthad, mean_squared_error(n,q))
##                                       
##for i in range(sig_ztautau.shape[0]):
##    t = sig_ztautau[i]
##    t = t.flatten()
##    w = sig_ztautau_preds[i]
##    w = w.flatten()                                                                            
##    errors_sig_ztautau = np.append(errors_sig_ztautau, mean_squared_error(t,w))


# ROC Curves

from sklearn.metrics import roc_curve

# assign binary labels - 0 for dijet events & 1 for signal events 
true_labels_zvvhbb = np.append(np.zeros(nomClusterTestData.shape[0]),np.ones(sig_zvvhbb.shape[0]))
true_labels_zprimetthad = np.append(np.zeros(nomClusterTestData.shape[0]),np.ones(sig_zprimetthad.shape[0]))
true_labels_ztautau = np.append(np.zeros(nomClusterTestData.shape[0]),np.ones(sig_ztautau.shape[0]))
true_labels_vbfhh = np.append(np.zeros(nomClusterTestData.shape[0]),np.ones(sig_vbfhh.shape[0]))
true_labels_ttbar = np.append(np.zeros(nomClusterTestData.shape[0]),np.ones(sig_ttbar.shape[0]))


# scale errors to simulate binary classification (is this valid?)
#max_zvhbb = np.amax(errors_sig_zvvhbb)
#max_zprimetthad = np.amax(errors_sig_zprimetthad)
#max_ztautau = np.amax(errors_sig_ztautau)

#scaled_errors_zvvhbb = np.append(errors_nomTestData,errors_sig_zvvhbb) / max_zvhbb
#scaled_errors_zprimetthad = np.append(errors_nomTestData,errors_sig_zprimetthad) / max_zprimetthad
#scaled_errors_ztautau = np.append(errors_nomTestData,errors_sig_ztautau) / max_ztautau

aa = np.append(errors_nomTestData,errors_sig_zvvhbb)
scaled2_errors_zvvhbb = aa / np.sqrt(np.sum(aa**2))

bb = np.append(errors_nomTestData,errors_sig_zprimetthad)
scaled2_errors_zprimetthad = bb / np.sqrt(np.sum(bb**2))

cc = np.append(errors_nomTestData,errors_sig_ztautau)
scaled2_errors_ztautau = cc / np.sqrt(np.sum(cc**2))

dd = np.append(errors_nomTestData,errors_sig_vbfhh)
scaled2_errors_vbfhh = dd / np.sqrt(np.sum(dd**2))

ee = np.append(errors_nomTestData,errors_sig_ttbar)
scaled2_errors_ttbar = ee / np.sqrt(np.sum(ee**2))


# with experiments.test():
fpr12, tpr12, thresholds12 = roc_curve(true_labels_zvvhbb, scaled2_errors_zvvhbb)
auc_12 = metrics.auc(fpr12,tpr12)
fpr22, tpr22, thresholds22 = roc_curve(true_labels_zprimetthad, scaled2_errors_zprimetthad)
auc_22 = metrics.auc(fpr22,tpr22)
fpr32, tpr32, thresholds32 = roc_curve(true_labels_ztautau, scaled2_errors_ztautau)
auc_32 = metrics.auc(fpr32,tpr32)

fpr42, tpr42, thresholds42 = roc_curve(true_labels_vbfhh, scaled2_errors_vbfhh)
auc_42 = metrics.auc(fpr42, tpr42)
fpr52, tpr52, thresholds52 = roc_curve(true_labels_ttbar, scaled2_errors_ttbar)
auc_52 = metrics.auc(fpr52, tpr52)


# np.save('/abvol/fpr/fpr (zvv,best).npy',fpr12)
# np.save('/abvol/tpr/tpr (zvv,best).npy',tpr12)

# np.save('/abvol/fpr/fpr (ztt,best).npy',fpr32)
# np.save('/abvol/tpr/tpr (ztt,best).npy',tpr32)

  #  metrics = {
  #      'zvv_AUC':auc_12,
  #      'ztt_AUC':auc_32
  #  }
  #  experiment.log_metrics(metrics)

#experiment.log_parameters(params)

plt.figure(figsize=(8,6))

plt.plot(fpr12,tpr12,label='zvhbb, AUC = ' + str(auc_12))
plt.plot(fpr22,tpr22,label='zprimetthad, AUC = ' + str(auc_22))
plt.plot(fpr32,tpr32,label='ztautau, AUC = ' + str(auc_32))
plt.plot(fpr42,tpr42,label='vbfhh, AUC = ' + str(auc_42))
plt.plot(fpr52,tpr52,label='ttbar, AUC = ' + str(auc_52))

 # FPR
plt.xscale(value='log')
plt.xlim(0.0001,1)

# TPR
plt.yscale(value='log')
plt.ylim(0.001,1)

plt.legend()
plt.savefig('/abvol/ROC_Curves/ROC Curve (best_model).png')

print(auc_12)
print(auc_22)
print(auc_32)
print(auc_42)
print(auc_52)

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


threshold_idx = find_nearest(fpr12,value=0.0001)
threshold = thresholds12[threshold_idx]

threshold_idx2 = find_nearest(fpr32,value=0.0001)
threshold2 = thresholds32[threshold_idx2]

threshold_idx3 = find_nearest(fpr42,value=0.0001)
threshold3 = thresholds42[threshold_idx3]

threshold_idx4 = find_nearest(fpr52,value=0.0001)
threshold4 = thresholds52[threshold_idx4]

# print(fpr12[threshold_idx])
# print(threshold)

scaled2_zvvhbb = scaled2_errors_zvvhbb[len(errors_nomTestData):]
scaled2_ztautau = scaled2_errors_ztautau[len(errors_nomTestData):]
scaled2_vbfhh = scaled2_errors_vbfhh[len(errors_nomTestData):]
scaled2_ttbar = scaled2_errors_ttbar[len(errors_nomTestData):]


indices = np.where(scaled2_zvvhbb > threshold)[0]
indices2 = np.where(scaled2_ztautau > threshold2)[0]
indices3 = np.where(scaled2_vbfhh > threshold3)[0]
indices4 = np.where(scaled2_ttbar > threshold4)[0]


print('zvvhbb: ' + str(len(indices)) + '/' + str(len(zvvhbb_met)))
print('ztautau: ' + str(len(indices2)) + '/' + str(len(ztautau_met)))
print('vbfhh: ' + str(len(indices3)) + '/' + str(len(vbfhh_met)))
print('ttbar: ' + str(len(indices4)) + '/' + str(len(ttbar_met)))



zvvhbb_met = zvvhbb_met[:,0]


valid_mets = zvvhbb_met[indices]


plt.figure(figsize=(8,6))
bin_vals_1,bins_1,_ = plt.hist(zvvhbb_met,bins=200,histtype='step',label='all events')
bin_vals_2,_,_ = plt.hist(valid_mets,bins=bins_1,histtype='step',label='events w/ disc > 10^-4')
plt.xlabel('met')
plt.ylabel('frequency')
plt.legend()
plt.savefig('met_efficiency.png')


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

plt.figure(figsize=(8,6))
plt.plot(bins1,divisors)
plt.legend()
plt.xlabel('met')
plt.ylabel('ratio')
plt.savefig('met_efficiency2.png')


#----------------------------------------------------------------------------------------

nomTestData_ptsum = []
zvvhbb_ptsum = []
zprimetthad_ptsum = []
ztautau_ptsum = []
ttbar_ptsum = []
vbfhh_ptsum = []


for i in range(sig_zvvhbb.shape[0]):
    a = sig_zvvhbb[i]
    zvvhbb_ptsum = np.append(zvvhbb_ptsum, np.sum(a[:,0]))

for i in range(sig_vbfhh.shape[0]):
    a = sig_vbfhh[i]
    vbfhh_ptsum = np.append(vbfhh_ptsum, np.sum(a[:,0]))

for i in range(sig_ttbar.shape[0]):
    a = sig_zvvhbb[i]
    ttbar_ptsum = np.append(ttbar_ptsum, np.sum(a[:,0]))




valid_pts = zvvhbb_ptsum[indices]

plt.figure(figsize=(8,6))
bin_vals_1,bins_1,_ = plt.hist(zvvhbb_ptsum,bins=200,histtype='step',label='all events')
bin_vals_2,_,_ = plt.hist(valid_pts,bins=bins_1,histtype='step',label='events w/ disc > 10^-4')
plt.xlabel('pT')
plt.ylabel('frequency')
plt.legend()
plt.savefig('/abvol/pt_efficiency/pt_efficiency (best,zvvbhbb).png')


valid_pts2 = vbfhh_ptsum[indices3]

plt.figure(figsize=(8,6))
bin_vals_3,bins_3,_ = plt.hist(vbfhh_ptsum,bins=200,histtype='step',label='all events')
bin_vals_4,_,_ = plt.hist(valid_pts2,bins=bins_3,histtype='step',label='events w/ disc > 10^-4')
plt.xlabel('pT')
plt.ylabel('frequency')
plt.legend()
plt.savefig('/abvol/pt_efficiency/pt_efficiency (best,vbfhh).png')

valid_pts3 = ttbar_ptsum[indices4]

plt.figure(figsize=(8,6))
bin_vals_5,bins_5,_ = plt.hist(ttbar_ptsum,bins=200,histtype='step',label='all events')
bin_vals_6,_,_ = plt.hist(valid_pts3,bins=bins_5,histtype='step',label='events w/ disc > 10^-4')
plt.xlabel('pT')
plt.ylabel('frequency')
plt.legend()
plt.savefig('/abvol/pt_efficiency/pt_efficiency (best,ttbar).png')


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


divisors1 = []

for i in range(bin_vals_3.shape[0]):
    if bin_vals_3[i] != 0:
        divisors1.append(bin_vals_2[i] / bin_vals_3[i])
    else:
        divisors1.append(0)

divisors1 = np.array(divisors1)
bins3 = []
for i in range(1, len(bins_3)):
    bins3.append(int((bins_3[i] + bins_3[i-1]) / 2))

bins3 = np.array(bins3)
divisors1 = np.array(divisors1)



divisors2 = []

for i in range(bin_vals_5.shape[0]):
    if bin_vals_5[i] != 0:
        divisors2.append(bin_vals_2[i] / bin_vals_5[i])
    else:
        divisors2.append(0)

divisors2 = np.array(divisors2)
bins5 = []
for i in range(1, len(bins_5)):
    bins5.append(int((bins_5[i] + bins_5[i-1]) / 2))

bins5 = np.array(bins5)
divisors2 = np.array(divisors2)

plt.figure(figsize=(8,6))
plt.plot(bins1,divisors)
plt.xlabel('pT')
plt.ylabel('ratio')
plt.legend()
plt.savefig('/abvol/pt_efficiency/pt_efficiency2 (best,zvvhbb).png')


plt.figure(figsize=(8,6))
plt.plot(bins3,divisors1)
plt.xlabel('pT')
plt.ylabel('ratio')
plt.legend()
plt.savefig('/abvol/pt_efficiency/pt_efficiency2 (best,vbfhh).png')

plt.figure(figsize=(8,6))
plt.plot(bins5,divisors2)
plt.xlabel('pT')
plt.ylabel('ratio')
plt.legend()
plt.savefig('/abvol/pt_efficiency/pt_efficiency2 (best,ttbar).png')
