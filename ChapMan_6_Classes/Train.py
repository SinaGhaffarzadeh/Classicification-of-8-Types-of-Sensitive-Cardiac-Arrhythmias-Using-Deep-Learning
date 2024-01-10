
"""

First part of training

"""

# Libraries 

import os
import math
import random
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import signal
from scipy.io import loadmat
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.utils import resample
from sklearn.preprocessing import LabelEncoder
from imblearn.datasets import make_imbalance
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_selection import RFE, RFECV
from imblearn.over_sampling import SMOTE

import tensorflow as tf
from tensorflow import keras
from keras.utils import plot_model
from keras.utils import pad_sequences

from biosppy.signals.tools import filter_signal

from functions import functions

# ------------------------------------------Reading existing files---------------------------------------------------- 
Dirction_and_labels = pd.read_excel('...\Direction_And_folds\Chap_CPSC_PTB_Direction_SingleLabels_CT-Code.xlsx')

with open('...\Direction_And_folds\Train_Test_Split_8Class_Chap_CPSC_PTB.pickle', 'rb') as handle:
    Folds_splited_data = pickle.load(handle)

""" This file is used in the training part of the Chapman dataset and  after that, we switch to the two above file  """
with open('...\Direction_And_folds\TestTrain_with_6fold_Chapman_dataset.pickle', 'rb') as handle:
    ChapMan_Rhythms = pickle.load(handle)

# ------------------------------------------Constant values and Empty lists-------------------------------------------

NumOfFold = 6
NumOfClass = 6
NumOfEpochs = 100
train_split = 0
test_split = 1
batchsize = 24


ChapMan_df = [] 
PAC_Rhythm = []
PVC_Rhythm = []
ChapMan_Without_PAC_PVC = [] 

# -------------------------------------------Preprocessing part-------------------------------------------------------

# Make a Data Frame with direction for K-fold of ChapMan dataset using Dirction_and_labels and ChapMan_Rhythms files
for fold in range(NumOfFold):
    Chap = []
    for TrTe in range(2):
        Chap_df_ = Dirction_and_labels.iloc[ChapMan_Rhythms[fold][TrTe]]
        Chap.append(Chap_df_)
    ChapMan_df.append([Chap[0],Chap[1]])


# Extract the index of PAC and PVC rhythms to remove them from our Data Frame
for fold in range(NumOfFold):
    PAC = []
    PVC = []
    for TrTe in range(2):

        PAC.append(ChapMan_df[fold][TrTe].Labs_Type[ChapMan_df[fold][TrTe].Labs_Type=='PAC'].index)
        PVC.append(ChapMan_df[fold][TrTe].Labs_Type[ChapMan_df[fold][TrTe].Labs_Type=='PVC'].index)
        
    PAC_Rhythm.append([PAC[0],PAC[1]])
    PVC_Rhythm.append([PVC[0],PVC[1]])

# Make train and test data frame without PVC and PAC
for fold in range(NumOfFold):
    Chap = []
    for TrTe in range(2):
        
        ChapMan_df_ = ChapMan_df[fold][TrTe].drop(PAC_Rhythm[fold][TrTe].tolist() + PVC_Rhythm[fold][TrTe].tolist())
        Chap.append(ChapMan_df_.set_index(pd.Index(np.arange(0,len(ChapMan_df_)))))
    
    ChapMan_Without_PAC_PVC.append([Chap[0],Chap[1]])

# Binarizing our k-fold labels
for fold in range(NumOfFold):
    
    for TrTe in range(2): 

        one_hot = functions.MultiLabelBinarizer()
        y_=one_hot.fit_transform(ChapMan_Without_PAC_PVC[fold][TrTe].CT_code.str.split(pat=','))
        print("The classes we will look at are encoded as SNOMED CT codes:")
        print(one_hot.classes_)
        y_1 = np.delete(y_, -1, axis=1)
        print("classes: {}".format(y_1.shape[1]))

        ChapMan_Without_PAC_PVC[fold][TrTe]['Labs'] = list(y_1)
        snomed_classes = one_hot.classes_[0:-1]


# ----------------------------------------------Training part--------------------------------------------------------

"""we used learning rate reducer, early stopping, and Checkpoint function in training process"""

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_Recall', factor=0.1, patience=1,
 verbose=1, mode='max', min_delta=0.0001, cooldown=0, min_lr=0)

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_Recall', mode='max', verbose=1, patience=2)


for i in range(NumOfFold):
    
    model = functions.residual_network_1d(6,trainable=True,trainable_last_layer=True,trainableOnelast=True,Classifire=1,LR=10e-3)
    
    fold = i

    checkpoint_filepath =f'.../Weights/Model_weights_fold_{fold}_6Class_ChapMan.hdf5'
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_Recall',
        mode='max',
        save_best_only=True)

    history = model.fit(x=functions.shuffle_batch_generator(batch_size=batchsize,
                                        gen_x=functions.generate_X_shuffle(np.asarray(ChapMan_Without_PAC_PVC[fold][0].Ecg_dir.tolist())),
                                        gen_y=functions.generate_y_shuffle(np.asarray(ChapMan_Without_PAC_PVC[fold][0].Labs.tolist())),snomed_classes=snomed_classes),
              epochs=NumOfEpochs, steps_per_epoch=(len(np.asarray(ChapMan_Without_PAC_PVC[fold][0].Ecg_dir.tolist()))/batchsize),
              validation_data=functions.generate_validation_data(np.asarray(ChapMan_Without_PAC_PVC[fold][1].Ecg_dir.tolist()),np.asarray(ChapMan_Without_PAC_PVC[fold][1].Labs.tolist())), 
              validation_freq=1, callbacks=[reduce_lr,early_stop,model_checkpoint_callback])





