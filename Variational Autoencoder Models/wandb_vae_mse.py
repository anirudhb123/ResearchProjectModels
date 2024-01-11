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


from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Input, Dense, GRU, Add, Concatenate, BatchNormalization, Conv2D, Conv2DTranspose, Lambda, Dot, Flatten, Layer, Reshape, MaxPooling1D,Dropout,MaxPooling2D,Conv2D,UpSampling2D,ReLU,LeakyReLU
from keras.models import Model, load_model

from tensorflow.keras import regularizers


from sklearn.metrics import mean_squared_error
from scipy.stats import wasserstein_distance

from wandb.keras import WandbCallback
wandb.init(project="Models", config={"key": "value"})


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


import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--label', action="store", dest="label", type=str, default="")
parser.add_argument('--nTrain', action="store", dest="nTrain", type=int, default=-1)
parser.add_argument('--nTotal', action="store", dest="nTotal", type=int, default=-1)
parser.add_argument('--nEpochs', action="store", dest="nEpochs", type=int, default=100) # change later but do 50 for now
parser.add_argument('--clusters', action="store", dest="clusters", type=int, default=100)
parser.add_argument('--batchSize', action="store", dest="batchSize", type=int, default=256)
parser.add_argument('--doAttention', action="store_true", dest="doAttention")
parser.add_argument('--useSig', action="store_true", dest="useSig")
parser.add_argument('--onlyPlot', action="store_true", dest="onlyPlot")

args = parser.parse_args()

clustersConsidered = args.clusters

numberOfEpochs = args.nEpochs
batchSize = args.batchSize

#print(args)

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


# shuffle both datasets in same way
p = np.random.permutation(len(nom_data))
nom_data = nom_data[p]
nom_met = nom_met[p]

nom_data = nom_data[:args.nTotal]
nom_data[:,:,:,0] = nom_data[:,:,:,0]/1000. # scaling from mega to giga


sig = np.concatenate([
    zvvhbb,
    ztautau
    ])

sig_met = np.concatenate([zvvhbb_met,ztautau_met])


nomTrainingDataLength = int(len(nom_data)*0.7)
nomValidationDataLength = int(len(nom_data)*0.1)
nomTestingDataLength = len(nom_data) - (nomTrainingDataLength + nomValidationDataLength)


nomClusterData = nom_data
training = nomClusterData[0:nomTrainingDataLength]
validation = nomClusterData[nomTrainingDataLength:nomTrainingDataLength + nomValidationDataLength]
testing = nomClusterData[-nomTestingDataLength:]

nomClusterTrainingData = np.array(training)
nomClusterValidationData = np.array(validation)
nomClusterTestData = np.array(testing)

nomMetTraining = nom_met[0:nomTrainingDataLength]
nomMetValidation = nom_met[nomTrainingDataLength:nomTrainingDataLength+nomValidationDataLength]
nomMetTesting = nom_met[-nomTestingDataLength:]


loss_arr_nom = []
loss_arr_zvvhbb = []
loss_arr_ztautau = []
loss_arr_test = []
val_weight = []


##def augment_data(data, num_augmentations):
##    augmented_data = []
##    for sample in data:
##        augmented_data.append(sample)
##        for _ in range(num_augmentations):
##            augmented_sample = sample + np.random.normal(loc=0, scale=0.5, size=sample.shape)
##            augmented_data.append(augmented_sample)
##    return np.array(augmented_data)
##
##
##num_augmentations = 1
##nomClusterTrainingData = augment_data(nomClusterTrainingData, num_augmentations)

def losses(data):
    appList = []
    for sample in data:
        appList.append(np.clip(1 - np.power(np.sum(sample[:,:,0].flatten()),4)/3000000000,a_min=0.00001,a_max=None))
    return appList

loss_arr_nom = np.array(losses(nomClusterValidationData))
val_weight = np.array(losses(nomClusterValidationData))
loss_arr_zvvhbb = np.array(losses(zvvhbb))
loss_arr_ztautau = np.array(losses(ztautau))


#nomClusterTrainingData = np.expand_dims(np.array(nomClusterTrainingData),axis=(3))
#nomClusterValidationData = np.expand_dims(np.array(nomClusterValidationData),axis=(3))
#nomClusterTestData = np.expand_dims(np.array(nomClusterTestData),axis=(3))

nomClusterTrainingLabels = nomClusterTrainingData.copy()
nomClusterValidationLabels = nomClusterValidationData.copy()
nomClusterTestLabels = nomClusterTestData.copy()

#def normalize(arr):
#    norm = np.linalg.norm(arr)
#    if norm == 0: 
#       return 1 * arr
#    return (1 * arr) / norm


#loss_arr_nom = normalize(np.array(loss_arr_nom))
#loss_arr_zvvhbb = normalize(np.array(loss_arr_zvvhbb))
#loss_arr_ztautau = normalize(np.array(loss_arr_ztautau))
#loss_arr_test = normalize(np.array(loss_arr_test))
#val_weight = normalize(np.array(val_weight))

#print('normalization reached')


from tensorflow.keras.regularizers import l2



## ---Try #2

input_cluster = Input(shape=(12, 12, 1), name='input_cluster')

x = Conv2D(16, 3, activation="relu", padding="same")(input_cluster)  # (12, 12, 16)
x = BatchNormalization()(x)
x = Conv2D(32, 3, activation="relu", padding="same")(x)  # (12, 12, 32)
x = BatchNormalization()(x)
x = MaxPooling2D(2, padding="same")(x)  # (6, 6, 32)
x = Conv2D(64, 3, activation="relu", padding="same")(x)  # (6, 6, 64)
x = BatchNormalization()(x)
x = MaxPooling2D(2, padding="same")(x)  # (3, 3, 64)
x = Flatten()(x)  # (576,)
x = Dense(288, activation="relu")(x)  # (288,)
x = Dense(96, activation="relu")(x)  # (96,)
z_mean = Dense(32, name='z_mean')(x)  # (32,)
z_log_var = Dense(32, name='z_log_var')(x)  # (32,)

def sampling(args):
    z_mean, z_log_var = args
    epsilon = tf.keras.backend.random_normal(shape=(tf.keras.backend.shape(z_mean)[0], 32), mean=0., stddev=1.)
    return z_mean + tf.keras.backend.exp(z_log_var * 0.5) * (epsilon + 0.000001)

z = Lambda(sampling, output_shape=(32,), name="z")([z_mean, z_log_var])  # (48,)
x = Dense(96, activation="relu")(z)  # (96,)
x = Dense(288, activation="relu")(x)  # (288,)
x = Dense(3 * 3 * 64, activation="relu")(x)  # (576,)
x = Reshape((3, 3, 64))(x)  # (3, 3, 64)
x = Conv2DTranspose(64, 3, strides=2, activation='relu',padding='same')(x) # (6,6,64)
x = Conv2DTranspose(32, 3, strides=1, activation="relu", padding="same")(x)  # (6, 6, 32)
x = Conv2DTranspose(32, 3, strides=2, activation="relu", padding="same")(x)  # (12, 12, 32)
x = Conv2DTranspose(16, 3, strides=1, activation="relu", padding="same")(x)  # (12, 12, 16)
output = Conv2D(1, 3, activation="relu", padding="same", name='output')(x)  # (12, 12, 1)



#print("Compiling")

model = Model(inputs=input_cluster, outputs= [output,z_mean,z_log_var])

print(model.summary())

# print(nomClusterTrainingData.shape)
# print(nomClusterTrainingLabels.shape)
# print(nomClusterValidationData.shape)
# print(nomClusterValidationLabels.shape)

emd_model = load_model("/abvol/cluster_models/EMDLossPred,model")


modelName = "mse_11"
fpr_weight = 1
r_factor = 1
kl_factor = 150

print(modelName)

def custom_loss1(y_true, y_pred):

    # reconstruction_loss = tf.math.square(abs(emd_model([y_true,y_true])))

    if tf.reduce_mean(y_pred[0] - y_true) < 0:
        reconstruction_loss = fpr_weight * tf.reduce_mean(tf.square(y_pred[0] - y_true))
    else:
        reconstruction_loss = tf.reduce_mean(tf.square(y_pred[0] - y_true))

    # Extract the z_mean and z_log_var from the output
    
    z_mean = y_pred[1]
    z_log_var = y_pred[2]

    # Compute the KL divergence
    kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))

    total_loss = kl_loss # reconstruction_loss * r_factor + kl_loss * kl_factor

    return total_loss



def custom_loss1(y_true, y_pred):
    reconstruction_loss = tf.reduce_mean(tf.square(y_pred[0] - y_true))
    
    z_mean = y_pred[1]
    z_log_var = y_pred[2]

    kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
    total_loss = reconstruction_loss * r_factor + kl_loss * kl_factor

    return total_loss



model.compile(optimizer='adam', loss=custom_loss1,weighted_metrics=[])
    
#print('Calculating')
    
modelCallbacks = [EarlyStopping(min_delta=0.001,patience=25),
                    ModelCheckpoint(filepath="/abvol/Pred1.hdf5", save_weights_only=True,
                                save_best_only=True)]

#with experiment.train():
history = model.fit(x=nomClusterTrainingData, y=nomClusterTrainingLabels, epochs=numberOfEpochs, batch_size=batchSize,
                callbacks=[WandbCallback()],
                validation_data=(nomClusterValidationData, nomClusterValidationLabels))


#print("Loading weights")
    
# model.load_weights("/abvol/Pred1.hdf5")

wandb.finish()


# model.save("/abvol/mse_models/"+modelName+",model")
    

for h in history.history:
    for ie in range(len(history.history[h])):
        history.history[h][ie] = float(history.history[h][ie])



#with open("/abvol/cluster_data/"+modelName+",history.json", "w") as f:
#    json.dump(history.history,f)


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


mses = []
emds = []
zvvhbb_mses = []
zvvhbb_emds = []
ztautau_mses = []
ztautau_emds = []


vbfhh_emds = []
ttbar_emds = []

vbfhh_mses = []
ttbar_mses = []



##def weighted_loss1(y_true, y_pred):
##    # y_true = y_true[:,:,:,0]
##    # y_pred = y_pred[:,:,:,0]
##
##    emds.append(abs(emd_model([y_true,y_pred[0]])))
##    mses.append(np.mean(np.square(y_true - y_pred[0])))
##
##    # reconstruction_loss = np.mean(np.square(y_true - y_pred[0]))
##    
##    grids = []
##    z_means = []
##    z_log_vars = []
##
##    for item in y_pred:
##        z_means.append(item[1])
##        z_log_vars.append(item[2])
##        grids.append(item[0])
##
##    grids = np.array(grids)
##    z_mean = np.array(z_means)
##    z_log_var = np.array(z_log_vars)
##    reconstruction_loss = tf.reduce_mean(np.square(y_true - grids))
##
##    
##    # z_mean = y_pred[1]
##    # z_log_var = y_pred[2]
##    
##    kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))    
##
##    kl_loss = tf.cast(kl_losses, dtype=tf.float64)
##    reconstruction_loss = tf.cast(reconstruction_losses, dtype=tf.float64)
##    
##    total_loss = np.add(reconstruction_loss, kl_loss)
##    
##    return total_loss


def weighted_loss(y_true, y_pred):
    # y_true = y_true[:,:,:,0]
    # y_pred = y_pred[:,:,:,0]

    zvvhbb_emds.append(abs(emd_model([y_true,y_pred])))
    zvvhbb_mses.append(np.mean(np.square(y_true - y_pred)))

    if np.mean((y_pred - y_true), axis=(0,1,2)) < 0:
        return np.mean(np.square(y_true - y_pred), axis=(1,2)) * r_factor * fpr_weight
    else:
        return np.mean(np.square(y_true - y_pred), axis=(1,2)) * r_factor


def weighted_loss2(z_log_var,z_mean):
    kl_loss = -0.5 * np.mean((1 + z_log_var - np.square(z_mean) - np.exp(z_log_var)), axis=1)
    return kl_loss # * kl_factor


##
##
##def weighted_loss3(y_true, y_pred,loss_array):
##    y_true = y_true[:,:,:,0]
##    y_pred = y_pred[:,:,:,0]
##
##    ztautau_emds.append(abs(emd_model([y_true,y_pred])))
##    ztautau_mses.append(np.mean(np.square(y_true - y_pred)))
##    return abs(emd_model([y_true,y_pred]))
##
##def vae_loss2(inputs, outputs, z_mean, z_log_var, loss_arr):
##    inputs_flat = tf.reshape(inputs, [-1, 12*12*1])
##    outputs_flat = tf.reshape(outputs, [-1, 12*12*1])
##    
##    reconstruction_loss = tf.reduce_mean(tf.square(inputs_flat - outputs_flat))
##    kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
##    
##    total_loss = reconstruction_loss + kl_loss
##    return total_loss


##print(nomClusterTestData.shape)
##print(model.predict(nomClusterTestData).shape)
##
##
##print(zvvhbb.shape)
##print(model.predict(zvvhbb).shape)
##
##
##print(ztautau.shape)
##print(model.predict(ztautau).shape)

mu = []
sigma = []

preds_to_avg = model.predict(nomClusterTestData)
preds_zvvhbb_to_avg = model.predict(zvvhbb)
preds_ztautau_to_avg = model.predict(ztautau)
preds_vbfhh_to_avg = model.predict(vbfhh)
preds_ttbar_to_avg = model.predict(ttbar)


num = 0

for i in range(num):
    preds_to_avg = preds_to_avg + model.predict(nomClusterTestData)
    preds_zvvhbb_to_avg = preds_zvvhbb_to_avg + model.predict(zvvhbb)
    preds_ztautau_to_avg = preds_ztautau_to_avg + model.predict(ztautau)
    preds_vbfhh_to_avg = preds_vbfhh_to_avg + model.predict(vbfhh)
    preds_ttbar_to_avg = preds_ttbar_to_avg + model.predict(ttbar)

nums = num + 1

preds = [i/nums for i in preds_to_avg]
preds_zvvhbb = [i/nums for i in preds_zvvhbb_to_avg]
preds_ztautau = [i/nums for i in preds_ztautau_to_avg]
preds_vbfhh = [i/nums for i in preds_vbfhh_to_avg]
preds_ttbar = [i/nums for i in preds_ttbar_to_avg]


mu_preds = np.array(preds[1])
for i in range(len(mu_preds)):
    mu.append(mu_preds[i])

sigma_preds = np.array(preds[2])
for i in range(len(sigma_preds)):
     sigma.append(sigma_preds[i])

mu = np.array(mu)
sigma = np.array(sigma)

mu = mu.flatten()
sigma = sigma.flatten()

# preds_zvvhbb = model.predict(zvvhbb)
# preds_ztautau = model.predict(ztautau)

errors_nomTestData = np.squeeze(weighted_loss(nomClusterTestData,preds[0]) + np.sum(weighted_loss2(preds[1],preds[2]), axis=0))

print('cluster shape: '+ str(nomClusterTestData.shape))
print('errors of cluster shape: ' + str(errors_nomTestData.shape))

errors_sig_zvvhbb = np.squeeze(weighted_loss(zvvhbb,preds_zvvhbb[0]) + np.sum(weighted_loss2(preds_zvvhbb[1],preds_zvvhbb[2]), axis=0))

print('zvvhbb cluster shape: '+ str(zvvhbb.shape))
print('errors of cluster shape: ' + str(errors_sig_zvvhbb.shape))

errors_sig_ztautau = np.squeeze(weighted_loss(ztautau,preds_ztautau[0]) + np.sum(weighted_loss2(preds_ztautau[1],preds_ztautau[2]), axis=0))

print('ztautau cluster shape: '+ str(ztautau.shape))
print('errors of cluster shape: ' + str(errors_sig_ztautau.shape))

errors_sig_vbfhh = np.squeeze(weighted_loss(vbfhh,preds_vbfhh[0]) + np.sum(weighted_loss2(preds_vbfhh[1],preds_vbfhh[2]), axis=0))
errors_sig_ttbar = np.squeeze(weighted_loss(ttbar,preds_ttbar[0]) + np.sum(weighted_loss2(preds_ttbar[1],preds_ttbar[2]), axis=0))



mses = np.array(mses)
emds = np.array(emds)
ztautau_mses = np.array(ztautau_mses)
ztautau_emds = np.array(ztautau_emds)
zvvhbb_mses = np.array(zvvhbb_mses)
zvvhbb_emds = np.array(zvvhbb_emds)

vbfhh_mses = np.array(vbfhh_mses)
ttbar_mses = np.array(ttbar_mses)

vbfhh_emds = np.array(vbfhh_emds)
ttbar_emds = np.array(ttbar_emds)

plt.figure(figsize=(8,6))
bins_x = np.linspace(0,700,num=200)
bin_vals_1,bins_1,_ = plt.hist(emds,bins=bins_x,histtype='step',label='all events (emd)')
bin_vals_2,_,_ = plt.hist(mses,bins=bins_1,histtype='step',label='all events (mse)')
plt.xlabel('distance')
plt.ylabel('frequency')
plt.legend()
plt.savefig('/abvol/comp_plots/mseVemd.png')

##
##plt.figure(figsize=(8,6))
##bins_x = np.linspace(0,700,num=200)
##bin_vals_1,bins_1,_ = plt.hist(zvvhbb_emds,bins=bins_x,histtype='step',label='all events (emd)')
##bin_vals_2,_,_ = plt.hist(zvvhbb_mses,bins=bins_1,histtype='step',label='all events (mse)')
##plt.xlabel('distance')
##plt.ylabel('frequency')
##plt.legend()
##plt.savefig('/abvol/comp_plots/mseVemd (zvvhbb).png')
##
##plt.figure(figsize=(8,6))
##bins_x = np.linspace(0,700,num=200)
##bin_vals_1,bins_1,_ = plt.hist(ztautau_emds,bins=bins_x,histtype='step',label='all events (emd)')
##bin_vals_2,_,_ = plt.hist(ztautau_mses,bins=bins_1,histtype='step',label='all events (mse)')
##plt.xlabel('distance')
##plt.ylabel('frequency')
##plt.legend()
##plt.savefig('/abvol/comp_plots/mseVemd (ztautau).png')






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

print(errors_sig_zvvhbb.shape)
print(true_labels_zvvhbb.shape)

print(errors_sig_ztautau.shape)
print(true_labels_ztautau.shape)


aa = np.append(errors_nomTestData,errors_sig_zvvhbb)
scaled2_errors_zvvhbb = aa / np.sqrt(np.sum(aa**2))

cc = np.append(errors_nomTestData,errors_sig_ztautau)
scaled2_errors_ztautau = cc / np.sqrt(np.sum(cc**2))

bb = np.append(errors_nomTestData,errors_sig_vbfhh)
scaled2_errors_vbfhh = bb / np.sqrt(np.sum(aa**2))

dd = np.append(errors_nomTestData,errors_sig_ttbar)
scaled2_errors_ttbar = dd / np.sqrt(np.sum(cc**2))


print(np.isnan(np.sum(scaled2_errors_zvvhbb)))
print(np.isnan(np.sum(scaled2_errors_ztautau)))


# with experiments.test():
fpr12, tpr12, thresholds12 = roc_curve(true_labels_zvvhbb, scaled2_errors_zvvhbb)
auc_12 = metrics.auc(fpr12,tpr12)
fpr32, tpr32, thresholds32 = roc_curve(true_labels_ztautau, scaled2_errors_ztautau)
auc_32 = metrics.auc(fpr32,tpr32)

fpr22, tpr22, thresholds22 = roc_curve(true_labels_vbfhh, scaled2_errors_vbfhh)
auc_22 = metrics.auc(fpr22, tpr22)
fpr42, tpr42, thresholds42 = roc_curve(true_labels_ttbar, scaled2_errors_ttbar)
auc_42 = metrics.auc(fpr42, tpr42)

##np.save('/abvol/fpr/fpr (zvv,vae_mse).npy',fpr12)
##np.save('/abvol/tpr/tpr (zvv,vae_mse).npy',tpr12)
##
##np.save('/abvol/fpr/fpr (ztt,vae_mse).npy',fpr32)
##np.save('/abvol/tpr/tpr (ztt,vae_mse).npy',tpr32)
##
##np.save('/abvol/fpr/fpr (vbfhh,vae_mse).npy',fpr22)
##np.save('/abvol/tpr/tpr (vbfhh,vae_mse).npy',tpr22)
##
##np.save('/abvol/fpr/fpr (ttbar,vae_mse).npy',fpr42)
##np.save('/abvol/tpr/tpr (ttbar,vae_mse).npy',tpr42)


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
plt.savefig('/abvol/ROC_Curves/ROC Curve (vae_mse).png')

print('zvvhbb AUC: ' + str(auc_12))
print('ztautau AUC: '+ str(auc_32))
print('vbfhh AUC: '+ str(auc_22))
print('ttbar AUC: '+ str(auc_42))

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
valid_mets = np.append(np.append(np.append(zvvhbb_met[indices_zvvhbb], ztautau_met[indices_ztautau]), vbfhh_met[indices_vbfhh]), ttbar_met[indices_ttbar])

plt.figure(figsize=(8,6))
bin_vals_1,bins_1,_ = plt.hist(mets,bins=200,histtype='step',label='all events')
bin_vals_2,_,_ = plt.hist(valid_mets,bins=bins_1,histtype='step',label='events w/ disc > 10^-4')
plt.xlabel('met')
plt.ylabel('frequency')
plt.xlim(0,300)
plt.legend()
plt.savefig('/abvol/eff_plots/met_efficiency (met) - all.png')

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
plt.savefig('/abvol/eff_plots/met_efficiency (met), log.png')


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
plt.savefig('/abvol/eff_plots/met_efficiency2 (mse).png')

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
plt.savefig('/abvol/eff_plots/met_efficiency2 (mse), log.png')

print('finished here')


#----------------------------------------------------------------------------------------

nomTestData_ptsum = np.array([])
zvvhbb_ptsum = np.array([])
zprimetthad_ptsum = np.array([])
ztautau_ptsum = np.array([])

def ptsum(arr):
    filtered_grids = []
    for grid in arr:
        filtered_grids.append(np.sum(grid[:,:,0]))
    return np.array(filtered_grids)

nom_ptsum = ptsum(nom_data)
zvvhbb_ptsum = ptsum(zvvhbb)
ztautau_ptsum = ptsum(ztautau)

valid_pts1 = np.array(zvvhbb_ptsum[indices_zvvhbb])
valid_pts2 = np.array(ztautau_ptsum[indices_ztautau])

ptsums = np.append(zvvhbb_ptsum,ztautau_ptsum)
valid_pts = np.append(valid_pts1,valid_pts2)


plt.figure(figsize=(8,6))
bin_vals_1,bins_1,_ = plt.hist(nom_ptsum,bins=200,histtype='step',label='background',density=True)
bin_vals_2,_,_ = plt.hist(zvvhbb_ptsum,bins=bins_1,histtype='step',label='zvvhbb',density=True)
bin_vals_2,_,_ = plt.hist(ztautau_ptsum,bins=bins_1,histtype='step',label='ztautau',density=True)
plt.xlabel('pT')
plt.ylabel('frequency')
plt.legend()
plt.savefig('/abvol/eff_plots/pT_distribution (mse).png')


plt.figure(figsize=(8,6))
bin_vals_1,bins_1,_ = plt.hist(ptsums,bins=200,histtype='step',label='all events')
bin_vals_2,_,_ = plt.hist(valid_pts,bins=bins_1,histtype='step',label='events w/ disc > 10^-4')
plt.xlabel('pT')
plt.ylabel('frequency')
plt.legend()
plt.savefig('/abvol/eff_plots/pt_efficiency (mse).png')

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
plt.xlabel('pT')
plt.ylabel('ratio')
plt.xlim(0,1000)
plt.legend()
plt.savefig('/abvol/eff_plots/pt_efficiency2 (mse).png')




#-------------------------------------Mu Distribution PLOTS---------------------------------------------------

plt.figure(figsize=(8,6))
plt.hist(mu,bins=2000,histtype='step',density=True,log=True)
plt.xlabel('mu')
plt.xlim(-0.05,0.15)
plt.ylabel('frequency')
plt.title('mu distribution')
plt.legend()
plt.savefig('/abvol/latent_plots/mu.png')

plt.figure(figsize=(8,6))
plt.hist(mu,bins=2000,histtype='step',density=True, log=True)
plt.xlabel('mu')
plt.xlim(-0.3,0.3)
plt.ylabel('frequency')
plt.title('mu distribution')
plt.legend()
plt.savefig('/abvol/latent_plots/mu_zvvhbb.png')

plt.figure(figsize=(8,6))
plt.hist(mu,bins=2000,histtype='step',density=True, log=True)
plt.xlabel('mu')
plt.xlim(-0.3,0.3)
plt.ylabel('frequency')
plt.title('mu distribution')
plt.legend()
plt.savefig('/abvol/latent_plots/mu_ztautau.png')


#-------------------------------------Sigma Distribution PLOTS---------------------------------------------------
plt.figure(figsize=(8,6))
plt.hist(sigma,bins=2000,histtype='step',density=True,log=True)
plt.xlabel('sigma')
plt.xlim(-0.05,0.15)
plt.ylabel('frequency')
plt.title('sigma distribution')
plt.legend()
plt.savefig('/abvol/latent_plots/sigma.png')

plt.figure(figsize=(8,6))
plt.hist(mu,bins=2000,histtype='step',density=True, log=True)
plt.xlabel('sigma (zvvhbb)')
plt.xlim(-0.3,0.3)
plt.ylabel('frequency')
plt.title('sigma distribution')
plt.legend()
plt.savefig('/abvol/latent_plots/sigma_zvvhbb.png')

plt.figure(figsize=(8,6))
plt.hist(mu,bins=2000,histtype='step',density=True, log=True)
plt.xlabel('mu')
plt.xlim(-0.3,0.3)
plt.ylabel('frequency')
plt.title('sigma distribution')
plt.legend()
plt.savefig('/abvol/latent_plots/sigma_ztautau.png')


print("done")

#-------------------------------------1D MSE PLOTS---------------------------------------------------


#test
plt.figure(figsize=(8,6))
plt.hist(errors_nomTestData,bins=200,histtype='step',density=True,range=[0,400])
plt.xlabel('mse')
plt.xlim(0,400)
plt.ylabel('frequency')
plt.title('background mse')
plt.legend()
plt.savefig('/abvol/mse_plots/background_mse.png')

#zvvhbb
plt.figure(figsize=(8,6))
plt.hist(errors_sig_zvvhbb,bins=200,histtype='step',density=True,range=[0,400])
plt.xlabel('mse')
plt.xlim(0,400)
plt.ylabel('frequency')
plt.title('zvvhbb mse')
plt.legend()
plt.savefig('/abvol/mse_plots/zvvhbb_mse.png')

#ztautau
plt.figure(figsize=(8,6))
plt.hist(errors_sig_ztautau,bins=200,histtype='step',density=True,range=[0,400])
plt.xlabel('mse')
plt.xlim(0,400)
plt.ylabel('frequency')
plt.title('ztautau mse')
plt.legend()
plt.savefig('/abvol/mse_plots/ztautau_mse.png')


#all mse's
plt.figure(figsize=(8,6))
plt.hist(errors_nomTestData,bins=200,histtype='step',density=True,range=[0,400],label='background')
plt.xlabel('mse')
plt.xlim(0,400)
plt.ylabel('frequency')
plt.title('mse')
plt.hist(errors_sig_zvvhbb,bins=200,histtype='step',density=True,range=[0,400],label='zvvhbb')
plt.hist(errors_sig_ztautau,bins=200,histtype='step',density=True,range=[0,400],label='ztautau')
plt.title('mse')
plt.legend()
plt.savefig('/abvol/mse_plots/all_mse.png')


print('mse finished')


#-------------------------------------2D Histogram PLOTS---------------------------------------------------


#zvvhbb
plt.figure(figsize=(8,6))
plt.hist2d(zvvhbb_ptsum,errors_sig_zvvhbb,range=[[0,200],[0,400]],bins=[25,50])
plt.xlabel('pT')
plt.xlim(0,200)
plt.ylim(0,400)
plt.ylabel('mse error')
plt.title('zvvhbb - pt vs. mse')
plt.legend()
plt.savefig('/abvol/2d_plots/zvvhbb_pt_mse.png')


plt.figure(figsize=(8,6))
plt.hist2d(zvvhbb_met,errors_sig_zvvhbb,range=[[0,200],[0,400]],bins=[25,50])
plt.xlabel('met')
plt.ylabel('mse error')
plt.xlim(0,200)
plt.ylim(0,400)
plt.title('zvvhbb - met vs. mse')
plt.legend()
plt.savefig('/abvol/2d_plots/zvvhbb_met_mse.png')


#ztautau

for i in range(ztautau.shape[0]):
    a = ztautau[i]
    ztautau_ptsum = np.append(ztautau_ptsum, np.sum(a[:,0]))

plt.figure(figsize=(8,6))
plt.hist2d(ztautau_ptsum,errors_sig_ztautau,range=[[0,200],[0,400]],bins=[25,50])
plt.xlabel('pT')
plt.xlim(0,200)
plt.ylim(0,400)
plt.ylabel('mse error')
plt.title('ztautau - pt vs. mse')
plt.legend()
plt.savefig('/abvol/2d_plots/ztautau_pt_mse.png')


ztautau_met = ztautau_met[:,0]

plt.figure(figsize=(8,6))
plt.hist2d(ztautau_met,errors_sig_ztautau,range=[[0,200],[0,400]],bins=[25,50])
plt.xlabel('met')
plt.ylabel('mse error')
plt.xlim(0,200)
plt.ylim(0,400)
plt.title('ztautau - met vs. mse')
plt.legend()
plt.savefig('/abvol/2d_plots/ztautau_met_mse.png')

#background

for i in range(nomClusterTestData.shape[0]):
    a = nomClusterTestData[i]
    nomTestData_ptsum = np.append(nomTestData_ptsum, np.sum(a[:,0]))

plt.figure(figsize=(8,6))
plt.hist2d(nomTestData_ptsum,errors_nomTestData,range=[[0,200],[0,400]],bins=[25,50])
plt.xlabel('pT')
plt.xlim(0,200)
plt.ylim(0,400)
plt.ylabel('mse error')
plt.title('background - pt vs. mse')
plt.legend()
plt.savefig('/abvol/2d_plots/test_pt_mse.png')

nomMetTesting = nomMetTesting[:,0]

plt.figure(figsize=(8,6))
plt.hist2d(nomMetTesting,errors_nomTestData,range=[[0,200],[0,400]],bins=[25,50])
plt.xlabel('met')
plt.ylabel('mse error')
plt.xlim(0,200)
plt.ylim(0,400)
plt.title('background - met vs. mse')
plt.legend()
plt.savefig('/abvol/2d_plots/test_met_mse.png')



#-------------------------------------1D met PLOTS---------------------------------------------------


#test
plt.figure(figsize=(8,6))
plt.hist(nomMetTesting,bins=25,histtype='step',density=True)
plt.xlabel('met')
plt.xlim(0,200)
plt.ylabel('frequency')
plt.title('background met')
plt.legend()
plt.savefig('/abvol/met_plots/background_met.png')

#zvvhbb
plt.figure(figsize=(8,6))
plt.hist(zvvhbb_met,bins=25,histtype='step',density=True)
plt.xlabel('met')
plt.xlim(0,200)
plt.ylabel('frequency')
plt.title('zvvhbb met')
plt.legend()
plt.savefig('/abvol/met_plots/zvvhbb_met.png')

#ztautau
plt.figure(figsize=(8,6))
plt.hist(ztautau_met,bins=25,histtype='step',density=True)
plt.xlabel('met')
plt.xlim(0,200)
plt.ylabel('frequency')
plt.title('ztautau met')
plt.legend()
plt.savefig('/abvol/met_plots/ztautau_met.png')

#all mets
plt.figure(figsize=(8,6))
plt.hist(nomMetTesting,bins=25,histtype='step',density=True,label='background')
plt.hist(zvvhbb_met,bins=25,histtype='step',density=True,label='zvvhbb')
plt.hist(ztautau_met,bins=25,histtype='step',density=True,label='ztautau')
plt.xlabel('met')
plt.xlim(0,200)
plt.ylabel('frequency')
plt.title('all met')
plt.legend()
plt.savefig('/abvol/met_plots/all_met.png')


#-------------------------------------1D pT PLOTS---------------------------------------------------


#test
plt.figure(figsize=(8,6))
plt.hist(nomTestData_ptsum,bins=25,histtype='step',density=True)
plt.xlabel('pT')
plt.xlim(0,200)
plt.ylabel('frequency')
plt.title('background pT')
plt.legend()
plt.savefig('/abvol/pt_plots/background_pt.png')

#zvvhbb
plt.figure(figsize=(8,6))
plt.hist(zvvhbb_ptsum,bins=25,histtype='step',density=True)
plt.xlabel('pT')
plt.xlim(0,200)
plt.ylabel('frequency')
plt.title('zvvhbb pT')
plt.legend()
plt.savefig('/abvol/pt_plots/zvvhbb_pt.png')

#ztautau
plt.figure(figsize=(8,6))
plt.hist(ztautau_ptsum,bins=25,histtype='step',density=True)
plt.xlabel('pT')
plt.xlim(0,200)
plt.ylabel('frequency')
plt.title('ztautau pT')
plt.legend()
plt.savefig('/abvol/pt_plots/ztautau_pt.png')

###zprimetthad
##plt.figure(figsize=(8,6))
##plt.hist(zprimetthad_ptsum,bins=25,histtype='step',density=True)
##plt.xlabel('pT')
##plt.xlim(0,200)
##plt.ylabel('frequency')
##plt.title('zprimetthad pT')
##plt.legend()
##plt.savefig('/abvol/pt_plots/zprimetthad_pt.png')


#all pts
plt.figure(figsize=(8,6))
plt.hist(nomTestData_ptsum,bins=25,histtype='step',density=True,label='background')
plt.hist(zvvhbb_ptsum,bins=25,histtype='step',density=True,label='zvvhbb')
plt.hist(ztautau_ptsum,bins=25,histtype='step',density=True,label='ztautau')
plt.xlabel('pT')
plt.xlim(0,200)
plt.ylabel('frequency')
plt.title('all pT')
plt.legend()
plt.savefig('/abvol/pt_plots/all_pt.png')

