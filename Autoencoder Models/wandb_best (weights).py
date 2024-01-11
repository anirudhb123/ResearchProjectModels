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

import wandb

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input, Dense, GRU, Add, Concatenate, BatchNormalization, Conv2D, Conv2DTranspose, Lambda, Dot, Flatten, Layer, Reshape, MaxPooling1D,Dropout,MaxPooling2D,Conv2D,UpSampling2D,ReLU,LeakyReLU
from keras.models import Model, load_model


from tensorflow.keras import regularizers



from sklearn.metrics import mean_squared_error
from scipy.stats import wasserstein_distance


from wandb.keras import WandbCallback
wandb.init(project="Models", config={"key": "value"})


# Opens files and reads data

clustersTotal = 200
entriesPerCluster = 3 # pt, eta, phi

import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--label', action="store", dest="label", type=str, default="")
parser.add_argument('--nTrain', action="store", dest="nTrain", type=int, default=-1)
parser.add_argument('--nTotal', action="store", dest="nTotal", type=int, default=-1)
parser.add_argument('--nEpochs', action="store", dest="nEpochs", type=int, default=100) # change later but do 50 for now
parser.add_argument('--clusters', action="store", dest="clusters", type=int, default=200)
parser.add_argument('--batchSize', action="store", dest="batchSize", type=int, default=1024)
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
nom_bkg_jz0 = np.load("/abvol/GepOutput_EM/jz0_pu200_grid.npy", 'r')
nom_bkg_jz1 = np.load("/abvol/GepOutput_EM/jz1_pu200_grid.npy", 'r')
nom_bkg_jz2 = np.load("/abvol/GepOutput_EM/jz2_pu200_grid.npy", 'r')
nom_sig_zvvhbb = np.load("/abvol/GepOutput_EM/zvvhbb_pu200_grid.npy", 'r')
nom_sig_ztautau = np.load("/abvol/GepOutput_EM/ztautau_pu200_grid.npy", 'r')
nom_sig_vbfhh = np.load("/abvol/GepOutput_EM/vbfhhbbbb_pu200_grid.npy", 'r')
nom_sig_ttbar = np.load("/abvol/GepOutput_EM/ttbar_pu200_grid.npy", 'r')

print(nom_bkg_jz0.shape)

nom_bkg_jz0 = nom_bkg_jz0[:,:,:,:1]
nom_bkg_jz1 = nom_bkg_jz1[:,:,:,:1]
nom_bkg_jz2 = nom_bkg_jz2[:,:,:,:1]
nom_sig_zvvhbb = nom_sig_zvvhbb[:,:,:,:1]
nom_sig_ztautau = nom_sig_ztautau[:,:,:,:1]
nom_sig_vbfhh = nom_sig_vbfhh[:,:,:,:1]
nom_sig_ttbar = nom_sig_ttbar[:,:,:,:1]

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

# shuffle both datasets in same way
p = np.random.permutation(len(nom_data))
nom_data = nom_data[p]
nom_met = nom_met[p]

print(nom_data.shape)
print(nom_met.shape)

nom_data = nom_data[:args.nTotal]
nom_data[:,:,:,0] = nom_data[:,:,:,0]/1000. # scaling from mega to giga


nomTrainingDataLength = int(len(nom_data)*0.7)
nomValidationDataLength = int(len(nom_data)*0.1)
nomTestingDataLength = len(nom_data) - (nomTrainingDataLength + nomValidationDataLength)


nomClusterData = nom_data
training = nomClusterData[0:nomTrainingDataLength]
validation = nomClusterData[nomTrainingDataLength:nomTrainingDataLength + nomValidationDataLength]
testing = nomClusterData[-nomTestingDataLength:]


nomMetTraining = nom_met[0:nomTrainingDataLength]
nomMetValidation = nom_met[nomTrainingDataLength:nomTrainingDataLength+nomValidationDataLength]
nomMetTesting = nom_met[-nomTestingDataLength:]

nomClusterTrainingData = training
nomClusterValidationData = validation
nomClusterTestData = testing



def augment_data(data, num_augmentations):
    augmented_data = []
    for sample in data:
        augmented_data.append(sample)
        for _ in range(num_augmentations):
            augmented_sample = sample + np.random.normal(loc=0, scale=0.01, size=sample.shape)
            augmented_data.append(augmented_sample)
    return np.array(augmented_data)

num_augmentations = 0
# nomClusterTrainingData = augment_data(nomClusterTrainingData, num_augmentations)


#nomClusterTrainingData = np.expand_dims(np.array(nomClusterTrainingData),axis=(3))
#nomClusterValidationData = np.expand_dims(np.array(nomClusterValidationData),axis=(3))
#nomClusterTestData = np.expand_dims(np.array(nomClusterTestData),axis=(3))

nomClusterTrainingLabels = nomClusterTrainingData.copy()
nomClusterValidationLabels = nomClusterValidationData.copy()
nomClusterTestLabels = nomClusterTestData.copy()

nomMetTraining = nomMetTraining[:,0].reshape(-1, 1)
nomMetValidation = nomMetValidation[:,0].reshape(-1, 1)
nomMetTesting = nomMetTesting[:,0].reshape(-1, 1)

print(nomMetTraining.shape)
print(nomClusterTrainingLabels.shape)


from tensorflow.keras.regularizers import l2

modelName = "Pred"

# Input
inputCluster = Input(shape=(24, 24, 1), name='inputCluster')

# Encoder
conv1 = Conv2D(32, (3, 3), padding='same')(inputCluster)
conv1 = BatchNormalization()(conv1)
conv1 = LeakyReLU()(conv1)
pool1 = MaxPooling2D((2, 2))(conv1) #(6,6,32)

conv2 = Conv2D(64, (3, 3), padding='same')(pool1)
conv2 = BatchNormalization()(conv2)
conv2 = LeakyReLU()(conv2)
pool2 = MaxPooling2D((2, 2))(conv2) #(6,6,64)

conv3 = Conv2D(64, (3, 3), padding='same')(pool2)
conv3 = BatchNormalization()(conv3)
conv3 = LeakyReLU()(conv3)
pool3 = MaxPooling2D((2, 2))(conv3) #(3,3,128)

# Latent Space
flatten = Flatten()(pool3)
# dense0 = Dense(576)(flatten)
dense1 = Dense(312)(flatten)
dense1 = LeakyReLU()(dense1)
dense2 = Dense(576)(dense1)
dense2 = ReLU()(dense2)
# dense3 = Dense(1152)(dense2)
# dense3 = ReLU()(dense3)

# Decoder
reshape = Reshape((3, 3, 64))(dense2)

conv_trans0 = Conv2DTranspose(64, (3, 3), padding='same')(reshape)
conv_trans0 = BatchNormalization()(conv_trans0)
conv_trans0 = LeakyReLU()(conv_trans0)
upsample0 = UpSampling2D((2, 2))(conv_trans0)

conv_trans1 = Conv2DTranspose(64, (3, 3), padding='same')(upsample0)
conv_trans1 = BatchNormalization()(conv_trans1)
conv_trans1 = LeakyReLU()(conv_trans1)
upsample1 = UpSampling2D((2, 2))(conv_trans1)

conv_trans2 = Conv2DTranspose(32, (3, 3), padding='same')(upsample1)
conv_trans2 = BatchNormalization()(conv_trans2)
conv_trans2 = LeakyReLU()(conv_trans2)
upsample2 = UpSampling2D((2, 2))(conv_trans2) 

conv_trans3 = Conv2DTranspose(1, (3, 3), padding='same')(upsample2)

output = conv_trans3


#print("Preparing Data")

clusterDataLength = clustersTotal


#print("Compiling")

model = Model(inputs=inputCluster, outputs=output)
    
print(model.summary())

print("Output Shape:", model.output_shape)


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


model.compile(optimizer='adam', loss=custom_loss,
              sample_weight_mode='temporal')
    
#print('Calculating')

##class CustomWandbCallback(WandbCallback):
##    def __init__(self, *args, **kwargs):
##        super().__init__(*args, **kwargs)
##    
##    def on_epoch_end(self, epoch, logs=None):
##        # Call the on_epoch_end method of the base callbacks
##        super(CustomWandbCallback, self).on_epoch_end(epoch, logs)
##        
##        # Log additional custom metrics to wandb
##        wandb.log({"custom_metric": logs["custom_metric"]})
##
### Initialize your custom WandbCallback
##custom_wandb_callback = CustomWandbCallback()
##    
##model_callbacks = [
##    EarlyStopping(min_delta=0.001, patience=25),
##    ModelCheckpoint(filepath="/abvol/Pred1.hdf5", save_weights_only=True, save_best_only=True),
##    custom_wandb_callback
##]

#with experiment.train():

nomMetTrainingLabels = nomMetTraining.copy()
nomMetValidationLabels = nomMetValidation.copy()
nomMetTestingLabels = nomMetTesting.copy()


def replace_values(arr,threshold):
    mask = arr > threshold
    result = np.zeros_like(arr)    
    result[mask] = 0
    result[~mask] = 1
    return result


nomMetTraining = replace_values(nomMetTraining,50)
nomMetValidation = replace_values(nomMetValidation,50)

print(nomMetTraining[0:50])
print(nomMetValidation[0:50])

history = model.fit(x=nomClusterTrainingData, y=nomClusterTrainingLabels,epochs=numberOfEpochs, batch_size=batchSize,
                callbacks=[WandbCallback()],
                validation_data=(nomClusterValidationData, nomClusterValidationLabels, nomMetValidation),sample_weight=nomMetTraining)
    
#print("Loading weights")
    
# model.load_weights("/abvol/Pred.hdf5")
    
#model.save("/abvol/cluster_models/"+modelName+",model")

wandb.finish()

for h in history.history:
    for ie in range(len(history.history[h])):
        history.history[h][ie] = float(history.history[h][ie])

#with open("/abvol/cluster_data/"+modelName+",history.json", "w") as f:
#    json.dump(history.history,f)



import matplotlib
import matplotlib.pyplot as plt



def weighted_loss(y_true, y_pred, weight_factor=fpr_weight):
    mse = np.square(y_true - y_pred)
    weighted_mse = np.where(y_pred > y_true, weight_factor * mse, mse)
    return np.mean(weighted_mse, axis=(1, 2, 3))

errors_nomTestData = weighted_loss(nomClusterTestData,model.predict(nomClusterTestData))
errors_sig_zvvhbb = weighted_loss(zvvhbb,model.predict(zvvhbb))
errors_sig_ztautau = weighted_loss(ztautau,model.predict(ztautau))
errors_sig_vbfhh = weighted_loss(vbfhh,model.predict(vbfhh))
errors_sig_ttbar = weighted_loss(ttbar,model.predict(ttbar))

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
true_labels_zvvhbb = np.append(np.zeros(nomClusterTestData.shape[0]),np.ones(zvvhbb.shape[0]))
true_labels_ztautau = np.append(np.zeros(nomClusterTestData.shape[0]),np.ones(ztautau.shape[0]))
true_labels_vbfhh = np.append(np.zeros(nomClusterTestData.shape[0]),np.ones(vbfhh.shape[0]))
true_labels_ttbar = np.append(np.zeros(nomClusterTestData.shape[0]),np.ones(ttbar.shape[0]))


# scale errors to simulate binary classification (is this valid?)
#max_zvhbb = np.amax(errors_sig_zvvhbb)
#max_zprimetthad = np.amax(errors_sig_zprimetthad)
#max_ztautau = np.amax(errors_sig_ztautau)

#scaled_errors_zvvhbb = np.append(errors_nomTestData,errors_sig_zvvhbb) / max_zvhbb
#scaled_errors_zprimetthad = np.append(errors_nomTestData,errors_sig_zprimetthad) / max_zprimetthad
#scaled_errors_ztautau = np.append(errors_nomTestData,errors_sig_ztautau) / max_ztautau

aa = np.append(errors_nomTestData,errors_sig_zvvhbb)
scaled2_errors_zvvhbb = aa / np.sqrt(np.sum(aa**2))

cc = np.append(errors_nomTestData,errors_sig_ztautau)
scaled2_errors_ztautau = cc / np.sqrt(np.sum(cc**2))

dd = np.append(errors_nomTestData,errors_sig_vbfhh)
scaled2_errors_vbfhh = dd / np.sqrt(np.sum(dd**2))

ee = np.append(errors_nomTestData,errors_sig_ttbar)
scaled2_errors_ttbar = ee / np.sqrt(np.sum(ee**2))


# with experiments.test():
fpr12, tpr12, thresholds12 = roc_curve(true_labels_zvvhbb, scaled2_errors_zvvhbb)
auc_12 = metrics.auc(fpr12,tpr12)
fpr32, tpr32, thresholds32 = roc_curve(true_labels_ztautau, scaled2_errors_ztautau)
auc_32 = metrics.auc(fpr32,tpr32)

fpr22, tpr22, thresholds22 = roc_curve(true_labels_vbfhh, scaled2_errors_vbfhh)
auc_22 = metrics.auc(fpr22, tpr22)
fpr42, tpr42, thresholds42 = roc_curve(true_labels_ttbar, scaled2_errors_ttbar)
auc_42 = metrics.auc(fpr42, tpr42)


##np.save('/abvol/fpr/fpr (zvv,best1).npy',fpr12)
##np.save('/abvol/tpr/tpr (zvv,best1).npy',tpr12)
##
##np.save('/abvol/fpr/fpr (ztt,best1).npy',fpr32)
##np.save('/abvol/tpr/tpr (ztt,best1).npy',tpr32)
##
##np.save('/abvol/fpr/fpr (vbfhh,best1).npy',fpr42)
##np.save('/abvol/tpr/tpr (vbfhh,best1).npy',tpr42)
##
##np.save('/abvol/fpr/fpr (ttbar,best1).npy',fpr52)
##np.save('/abvol/tpr/tpr (ttbar,best1).npy',tpr52)

  #  metrics = {
  #      'zvv_AUC':auc_12,
  #      'ztt_AUC':auc_32
  #  }
  #  experiment.log_metrics(metrics)

#experiment.log_parameters(params)

plt.figure(figsize=(8,6))

plt.plot(fpr12,tpr12,label='zvhbb, AUC = ' + str(auc_12))
plt.plot(fpr32,tpr32,label='ztautau, AUC = ' + str(auc_32))
plt.plot(fpr22,tpr22,label='vbfhh, AUC = ' + str(auc_22))
plt.plot(fpr42,tpr42,label='ttbar, AUC = ' + str(auc_42))

 # FPR
plt.xscale(value='log')
plt.xlim(0.0001,1)

# TPR
plt.yscale(value='log')
plt.ylim(0.001,1)

plt.legend()
plt.savefig('/abvol/ROC_Curves/ROC Curve (best_model).png')

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


threshold_idx = find_nearest(fpr12,value=0.0001)
threshold = thresholds12[threshold_idx]

# print(str(fpr12[threshold_idx]) + " / " + str(threshold))


threshold_idx2 = find_nearest(fpr12,value=0.001)
threshold2 = thresholds12[threshold_idx2]
# print(str(fpr12[threshold_idx2]) + " / " + str(threshold))

threshold_idx3 = find_nearest(fpr32,value=0.0001)
threshold3 = thresholds32[threshold_idx3]

# print(str(fpr12[threshold_idx]) + " / " + str(threshold))


threshold_idx4 = find_nearest(fpr32,value=0.001)
threshold4 = thresholds32[threshold_idx4]
# print(str(fpr12[threshold_idx2]) + " / " + str(threshold))




threshold_idx5 = find_nearest(fpr22,value=0.0001)
threshold5 = thresholds22[threshold_idx5]

threshold_idx6 = find_nearest(fpr22,value=0.001)
threshold6 = thresholds22[threshold_idx6]


threshold_idx7 = find_nearest(fpr42,value=0.0001)
threshold7 = thresholds42[threshold_idx7]


threshold_idx8 = find_nearest(fpr42,value=0.001)
threshold8 = thresholds42[threshold_idx8]


#----------------------------------------------------------------------------------------

scaled2_zvvhbb = scaled2_errors_zvvhbb[len(errors_nomTestData):]

indices_zvvhbb = np.where(scaled2_zvvhbb > threshold)[0]

print('zvvhbb (0.001): ' + str(len(indices_zvvhbb)) + " / " + str(len(zvvhbb_met)))

indices_zvvhbb_2 = np.where(scaled2_zvvhbb > threshold2)[0]

print('zvvhbb (0.01): ' + str(len(indices_zvvhbb_2)) + " / " + str(len(zvvhbb_met)))

zvvhbb_met = zvvhbb_met[:,0]



scaled2_ztautau = scaled2_errors_ztautau[len(errors_nomTestData):]

indices_ztautau = np.where(scaled2_ztautau > threshold3)[0]

print('ztautau (0.001): ' + str(len(indices_ztautau)) + " / " + str(len(ztautau_met)))

indices_ztautau_2 = np.where(scaled2_ztautau > threshold4)[0]

print('ztautau (0.01): ' + str(len(indices_ztautau_2)) + " / " + str(len(ztautau_met)))

ztautau_met = ztautau_met[:,0]



scaled2_vbfhh = scaled2_errors_vbfhh[len(errors_nomTestData):]

indices_vbfhh = np.where(scaled2_vbfhh > threshold5)[0]

print('vbfhh (0.001): ' + str(len(indices_vbfhh)) + " / " + str(len(vbfhh_met)))

indices_vbfhh_2 = np.where(scaled2_vbfhh > threshold6)[0]

print('vbfhh (0.01): ' + str(len(indices_vbfhh_2)) + " / " + str(len(vbfhh_met)))

vbfhh_met = vbfhh_met[:,0]


scaled2_ttbar = scaled2_errors_ttbar[len(errors_nomTestData):]

indices_ttbar = np.where(scaled2_ttbar > threshold7)[0]

print('ttbar (0.001): ' + str(len(indices_ttbar)) + " / " + str(len(ttbar_met)))

indices_ttbar_2 = np.where(scaled2_ttbar > threshold8)[0]

print('ttbar (0.01): ' + str(len(indices_ttbar_2)) + " / " + str(len(ttbar_met)))

ttbar_met = ttbar_met[:,0]



mets = np.append(np.append(np.append(zvvhbb_met,ztautau_met), vbfhh_met), ttbar_met)
valid_mets = np.append(np.append(np.append(zvvhbb_met[indices_zvvhbb_2], ztautau_met[indices_ztautau_2]), vbfhh_met[indices_vbfhh_2]), ttbar_met[indices_ttbar_2])

plt.figure(figsize=(8,6))
bin_vals_1,bins_1,_ = plt.hist(mets,bins=200,histtype='step',label='all events')
bin_vals_2,_,_ = plt.hist(valid_mets,bins=bins_1,histtype='step',label='events w/ disc > 10^-4')
plt.xlabel('met')
plt.ylabel('frequency')
plt.xlim(0,300)
plt.legend()
plt.savefig('/abvol/eff_plots/met_efficiency (best, weighted) - all.png')

#all 
plt.figure(figsize=(8,6))
bin_vals_all_1,bins_all_1,_ = plt.hist(mets,bins=200,histtype='step',label='all events')
bin_vals_all_2,_,_ = plt.hist(valid_mets,bins=bins_all_1,histtype='step',label='events w/ disc > 10^-4')
plt.xlabel('met')
plt.ylabel('frequency')
plt.xlim(0,300)
plt.yscale('log')
plt.legend()
# plt.savefig('/abvol/eff_plots/met_efficiency (emd), log.png')

#zvvhbb
plt.figure(figsize=(8,6))
bin_vals_zvv_1,bins_zvv_1,_ = plt.hist(zvvhbb_met,bins=200,histtype='step',label='all events')
bin_vals_zvv_2,_,_ = plt.hist(zvvhbb_met[indices_zvvhbb],bins=bins_zvv_1,histtype='step',label='events w/ disc > 10^-4')
plt.xlabel('met')
plt.ylabel('frequency')
plt.xlim(0,300)
plt.yscale('log')
plt.legend()
# plt.savefig('/abvol/eff_plots/met_efficiency (emd) - zvvhbb, log.png')

#ztt
plt.figure(figsize=(8,6))
bin_vals_ztt_1,bins_ztt_1,_ = plt.hist(ztautau_met,bins=200,histtype='step',label='all events')
bin_vals_ztt_2,_,_ = plt.hist(ztautau_met[indices_ztautau],bins=bins_ztt_1,histtype='step',label='events w/ disc > 10^-4')
plt.xlabel('met')
plt.ylabel('frequency')
plt.xlim(0,300)
plt.yscale('log')
plt.legend()
# plt.savefig('/abvol/eff_plots/met_efficiency (emd), log.png')

#vbfhh 
plt.figure(figsize=(8,6))
bin_vals_vbfhh_1,bins_vbfhh_1,_ = plt.hist(vbfhh_met,bins=200,histtype='step',label='all events')
bin_vals_vbfhh_2,_,_ = plt.hist(vbfhh_met[indices_vbfhh],bins=bins_vbfhh_1,histtype='step',label='events w/ disc > 10^-4')
plt.xlabel('met')
plt.ylabel('frequency')
plt.xlim(0,300)
plt.yscale('log')
plt.legend()
# plt.savefig('/abvol/eff_plots/met_efficiency (emd), log.png')

#ttbar
plt.figure(figsize=(8,6))
bin_vals_ttbar_1,bins_ttbar_1,_ = plt.hist(ttbar_met,bins=200,histtype='step',label='all events')
bin_vals_ttbar_2,_,_ = plt.hist(ttbar_met[indices_ttbar],bins=bins_ttbar_1,histtype='step',label='events w/ disc > 10^-4')
plt.xlabel('met')
plt.ylabel('frequency')
plt.xlim(0,300)
plt.yscale('log')
plt.legend()
plt.savefig('/abvol/eff_plots/met_efficiency (best, weighted), log.png')


# all 
divisors_all = []

for i in range(bin_vals_all_1.shape[0]):
    if bin_vals_all_1[i] != 0:
        divisors_all.append(bin_vals_all_2[i] / bin_vals_all_1[i])
    else:
        divisors_all.append(0)

divisors_all = np.array(divisors_all)
bins1_all = []
for i in range(1,len(bins_all_1)):
    bins1_all.append(int((bins_all_1[i] + bins_all_1[i-1]) / 2))

bins1_all = np.array(bins1_all)
divisors_all = np.array(divisors_all)

# zvvhbb
divisors_zvv = []

for i in range(bin_vals_zvv_1.shape[0]):
    if bin_vals_zvv_1[i] != 0:
        divisors_zvv.append(bin_vals_zvv_2[i] / bin_vals_zvv_1[i])
    else:
        divisors_zvv.append(0)

divisors_zvv = np.array(divisors_zvv)
bins1_zvv = []
for i in range(1, len(bins_zvv_1)):
    bins1_zvv.append(int((bins_zvv_1[i] + bins_zvv_1[i - 1]) / 2))

bins1_zvv = np.array(bins1_zvv)
divisors_zvv = np.array(divisors_zvv)

# ztt
divisors_ztt = []

for i in range(bin_vals_ztt_1.shape[0]):
    if bin_vals_ztt_1[i] != 0:
        divisors_ztt.append(bin_vals_ztt_2[i] / bin_vals_ztt_1[i])
    else:
        divisors_ztt.append(0)

divisors_ztt = np.array(divisors_ztt)
bins1_ztt = []
for i in range(1, len(bins_ztt_1)):
    bins1_ztt.append(int((bins_ztt_1[i] + bins_ztt_1[i - 1]) / 2))

bins1_ztt = np.array(bins1_ztt)
divisors_ztt = np.array(divisors_ztt)

# vbfhh
divisors_vbfhh = []

for i in range(bin_vals_vbfhh_1.shape[0]):
    if bin_vals_vbfhh_1[i] != 0:
        divisors_vbfhh.append(bin_vals_vbfhh_2[i] / bin_vals_vbfhh_1[i])
    else:
        divisors_vbfhh.append(0)

divisors_vbfhh = np.array(divisors_vbfhh)
bins1_vbfhh = []
for i in range(1, len(bins_vbfhh_1)):
    bins1_vbfhh.append(int((bins_vbfhh_1[i] + bins_vbfhh_1[i - 1]) / 2))

bins1_vbfhh = np.array(bins1_vbfhh)
divisors_vbfhh = np.array(divisors_vbfhh)

# ttbar
divisors_ttbar = []


for i in range(bin_vals_ttbar_1.shape[0]):
    if bin_vals_ttbar_1[i] != 0:
        divisors_ttbar.append(bin_vals_ttbar_2[i] / bin_vals_ttbar_1[i])
    else:
        divisors_ttbar.append(0)

divisors_ttbar = np.array(divisors_ttbar)
bins1_ttbar = []
for i in range(1, len(bins_ttbar_1)):
    bins1_ttbar.append(int((bins_ttbar_1[i] + bins_ttbar_1[i - 1]) / 2))

bins1_ttbar = np.array(bins1_ttbar)
divisors_ttbar = np.array(divisors_ttbar)


plt.figure(figsize=(8,6))
plt.plot(bins1_zvv, divisors_zvv, label='zvv')
plt.plot(bins1_ztt, divisors_ztt, label='ztt')
plt.plot(bins1_vbfhh, divisors_vbfhh, label='vbfhh')
plt.plot(bins1_ttbar, divisors_ttbar, label='ttbar')
plt.legend()
plt.xlabel('met')
plt.ylabel('ratio')
plt.xlim(0,300)
plt.savefig('/abvol/eff_plots/met_efficiency2 (best, high removed).png')

plt.figure(figsize=(8,6))
plt.yscale('log')
plt.plot(bins1_zvv, divisors_zvv, label='zvv')
plt.plot(bins1_ztt, divisors_ztt, label='ztt')
plt.plot(bins1_vbfhh, divisors_vbfhh, label='vbfhh')
plt.plot(bins1_ttbar, divisors_ttbar, label='ttbar')
plt.legend()
plt.xlabel('met')
plt.ylabel('ratio')
plt.xlim(0,300)
plt.savefig('/abvol/eff_plots/met_efficiency2 (best, weighted), log.png')

print('finished here')



#----------------------------------------------------------------------------------------

nomTestData_ptsum = []
zvvhbb_ptsum = []
ztautau_ptsum = []
ttbar_ptsum = []
vbfhh_ptsum = []


for i in range(zvvhbb.shape[0]):
    a = zvvhbb[i]
    zvvhbb_ptsum = np.append(zvvhbb_ptsum, np.sum(a[:,:,0]))

for i in range(vbfhh.shape[0]):
    a = vbfhh[i]
    vbfhh_ptsum = np.append(vbfhh_ptsum, np.sum(a[:,:,0]))

for i in range(ttbar.shape[0]):
    a = ttbar[i]
    ttbar_ptsum = np.append(ttbar_ptsum, np.sum(a[:,:,0]))



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
