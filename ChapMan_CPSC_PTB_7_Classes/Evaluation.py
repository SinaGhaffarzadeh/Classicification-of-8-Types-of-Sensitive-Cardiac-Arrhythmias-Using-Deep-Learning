

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
from sklearn.metrics import classification_report

from functions import functions


NumOfFold = 6
NumOfClass = 7

model = functions.residual_network_1d(NumOfClass,trainable=True,trainable_last_layer=True,trainableOnelast=True,Classifire=1,LR=10e-3)


# General report of classification of each folds
for fold in range(NumOfFold):
    
    model.load_weights(f'...\ChapMan_CPSC_PTB_7_Classes\Weights\Model_weights_{fold}_7Class_Chap_CPSC_PTB_with_PAC.hdf5')
    
    print(f'Fold {fold}')
    prediction = model.predict(functions.generate_validation_data(np.asarray(Chap_CPSC_PTB_df_Without_PVC[fold][1].Ecg_dir.tolist()),
        np.asarray(Chap_CPSC_PTB_df_Without_PVC[fold][1].Labs.tolist()))[0], batch_size=32)
    a = [ 'Afib','Af','PAC','SB','SVT','SIN','STach' ]
    seven_class_valu = np.asarray(Chap_CPSC_PTB_df_Without_PVC[fold][1].Labs.tolist())
    print(classification_report(np.argmax(seven_class_valu,axis=1), np.argmax(prediction,axis=1), target_names=a),'\n')


# Confusion Matrix of each folds
"""

According to this that, our data is imbalanced,
and also to have a good analysis we have to report them on the balance form.
So, the below code able to carry out this processing 

below code for carrying out this process, first of all,
find the length of the smallest class and then divide the other classes into
bunchs of the smallest class length and find a confusion matrix for each bunch 
and ultimately, the mean of these confusion matrices is reported.


"""
for fold in range(NumOfFold):
    
    model.load_weights(f'...\ChapMan_CPSC_PTB_7_Classes\Weights\Model_weights_{fold}_7Class_Chap_CPSC_PTB_with_PAC.hdf5')
    
    print(f'Fold {fold}')
    y_pred = model.predict(functions.generate_validation_data(np.asarray(Chap_CPSC_PTB_df_Without_PVC[fold][1].Ecg_dir.tolist()),
    	np.asarray(Chap_CPSC_PTB_df_Without_PVC[fold][1].Labs.tolist()))[0], batch_size=32)
    a = ['Afib','Af','PAC','SB','SVT','SIN','STach']
    
    actual = np.asarray(Chap_CPSC_PTB_df_Without_PVC[fold][1].Labs.tolist())

    c1 = []
    c2 = []
    c3 = []
    c4 = []
    c5 = []
    c6 = []
    c7 = []

    seven_class = []
    for i in range(actual.shape[0]):
            if actual[i,0] == 1:
                c1.append(i)
            if actual[i,1] == 1:
                c2.append(i)
            if actual[i,2] == 1:
                c3.append(i)
            if actual[i,3] == 1:
                c4.append(i)
            if actual[i,4] == 1:
                c5.append(i)
            if actual[i,5] == 1:
                c6.append(i)
            if actual[i,6] == 1:
                c7.append(i)

    for i in range(actual.shape[0]):
        for g in range(7):
            if actual[i,g] == 0:
                pass
            else:
                seven_class.append(i)


    classes = [c1,c2,c3,c4,c5,c6,c7]
    classes_len = []

    for i in range(len(classes)):
        classes_len.append(len(classes[i]))
    classes_len

    min_num_class = min(classes_len)
    min_num_class_index = classes_len.index(min_num_class)

    devs = []
    for i in range(7):
        d = classes_len[i]//min_num_class
        if d == 1:
            devs.append(classes_len[i]/min_num_class)
        else:
            devs.append(d)
    print(devs)

    class_arr = np.array(classes[4])
    dev_class = []
    import random
    for i in range(len(devs)):
        if devs[i]>=2:
            dev_class.append(np.random.randint(classes_len[i], size=(devs[i],min_num_class)))
        if devs[i] == 1:
            dev_class.append(np.arange(0,min_num_class).reshape(1,min_num_class))
        if 1 < devs[i] < 2 :
            dev_class.append(np.random.randint(classes_len[i], size=(1,min_num_class)))
    len(dev_class)
    for i in range(len(dev_class)):
        print(dev_class[i].shape)

    import numpy as np
    from sklearn import metrics

    mat_devd = []

    sup_mat = np.array([[0,0,0,0,0,0,1],[0,0,0,0,0,0,1],[0,0,0,0,0,0,1],[0,0,0,0,0,0,1],
                       [0,0,0,0,0,0,1],[0,0,0,0,0,0,1],[0,0,0,0,0,0,1]])

    sup_mat_2 = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0],
                       [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])


    conf_mat = np.zeros((7,7))

    for g in range(7):
        cm = np.zeros((dev_class[g].shape[0],7,7))

        for i in range(dev_class[g].shape[0]):

            pred_without_undifined = y_pred[np.array(classes[g])[dev_class[g][i,:]]]
            pred_without_undifined_dif = y_pred[np.array(classes[-1])[dev_class[-1]]]

            conc_pred = np.concatenate((pred_without_undifined,sup_mat_2),axis=0)

            actual_without_undifined = actual[np.array(classes[g])[dev_class[g][i,:]]]
            actual_without_undifined_dif = actual[np.array(classes[-1])[dev_class[-1]]]
            conc_actual = np.concatenate((actual_without_undifined,sup_mat),axis=0)

            mat_devd.append([conc_pred,conc_actual])

            y_preed=np.argmax(conc_pred, axis=1)
            y_test=np.argmax(conc_actual, axis=1)
            cm[i,:,:] = metrics.confusion_matrix(y_test, y_preed)

            if g==min_num_class_index:
                cm = cm - 1

        cm_mean = np.rint(cm.mean(axis=0))
        conf_mat[g,:] = cm_mean[g,:]

    print(conf_mat)
    print(len(mat_devd),',',cm.shape,',',cm_mean.shape)

    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = conf_mat, display_labels = a)

    cm_display.plot()
    plt.show()