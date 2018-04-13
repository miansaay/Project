#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 19:21:23 2017

@author: obarquero
"""
import scipy.io as io
import wfdb
import numpy as np
import scipy.signal as sg
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from imblearn.over_sampling import SMOTE


def read_header(fname,path):
    """
    Funcion para leer el header de las señales
    """
    
    idx_point = fname.find('.')
    fname = fname[:idx_point]
    header = wfdb.rdheader(path+fname)
    
    return header
 

#Función para leer las señales

def read_challenge_mat_files(fname,path):
    """
    Lee las señales .mat del challenge 2017 y devuelve un array de numpy

    Input
    ------
       .- fname: nombre del fichero para leer

    Output
    ------
       .- ecg: numpy array con los valores de ecg
       .- header: dicccionario con la información del header
    """

    #get only the id, remove the extension file
    #fname_header = fname
    
    #header = read_header(fname_header)

    #read mat file
    header = read_header(fname,path)

    mat_file = io.loadmat(path+fname)

    #mat_file is a dictionary, and the data is on the 'var' key

    ecg = mat_file['val'].flatten()

    return ecg,header.__dict__

def processing_ecg(ecg,fs = 300):
    """
    Function to filter ecg. We are going to make a simple filter between 
    
    """
    
    order = int(0.3*fs)
    fnyq = fs/2.
    fc = [2/fnyq,45/fnyq] #frecuencias de corte 2Hz y 45Hz 
    
    
    a,b =np.array([1]), sg.firwin(order,fc,pass_zero = False) 
    ecg_filtered = sg.filtfilt(b, a, ecg)
    
    return ecg_filtered

def get_class(header):
    """
    Function that given a header from a record return the class it belongs
    """
    
    classes = np.loadtxt('./physionet_challenge/training2017/REFERENCE.csv',delimiter = ',',dtype = 'str')
    
    ids = list(classes[:,0])
    classes = list(classes[:,1])
    
    id_h = header['recordname'] #get recordname
    
    #TO DO? Control if there exist id_h in ids
    idx = ids.index(id_h) #get idx to obtain class

    #return class for the current record
    return classes[idx]
    

def plot_all_records(plot_original = True):
    """
    Function that plot all records
    """
    
    curr_dir = os.getcwd()
    #os.chdir('./physionet_challenge/training2017/')
    path = './physionet_challenge/training2017/'
    
    #TO DO
    #implement some mechanism to abort execution
    #for over .mat
    for fname in list(glob.glob(path+'*.mat')):
        
        #read ecg
        ecg,header = read_challenge_mat_files(os.path.basename(fname),path)
        
        #function to get the the class
        class_ecg = get_class(header)
        thesaurus = {'N':'Normal','A':'AF','O':'Other Rhyth','~':'Noisy'}
        
        #plot current ecg
        t = np.arange(len(ecg))/float(header['fs'])
        
        #preprocessing ecg
        ecg_filtered = processing_ecg(ecg,fs = 300)
        tit = 'ECG. Record: '+ header['recordname'] + ' Class: '+ thesaurus[class_ecg]
        
        plt.close('all')
        
        if plot_original:
            plt.plot(t,ecg,label = 'ecg original')
            
        plt.plot(t,ecg_filtered,label = 'ecg_filtered')
        plt.xlabel('Time (sec)')
        plt.title(tit)
        plt.ylabel('Amplitude (mV)')
        plt.axis('tight')
        
        #plt.waitforbuttonpress() 
                    
def count(sequence):
    
    keys, counts = np.unique(sequence, return_counts=True)
    counts = dict(zip(keys, counts))
    
    return counts
    
    
def get_distribution_classes():
    """
    Function to get the distribution of differente clases
    """
    #MA tu código aquí
    
    #leer references.csv y recontar el número de cada clase
    
    classes = np.loadtxt('./physionet_challenge/training2017/REFERENCE.csv',delimiter = ',',dtype = 'str')
    classes = list(classes[:,1])
    count_classes = count(classes)
   
    return count_classes
  
def get_distribution_length():
    """
    Function to get a distribution of the signal lengths
    """
    #MA tu código aquí
    
    #hacer un for sobre todos los datos, obtener la longitud de cada señal y guardarla en una lista
    path = './physionet_challenge/training2017/'
    
    length_ecg = []
    
    for fname in list(glob.glob(path+'*.mat')):
        ecg,header = read_challenge_mat_files(os.path.basename(fname),path)
        length_ecg.append(header['siglen'])
    
    plt.hist(length_ecg, 10)
    plt.xlabel('ECGs lengths')
    plt.grid(True)
    plt.show()
    
    return length_ecg

def get_distribution_less9000():
    """
    Function to get distribution by classes for signals with less than 9000
    """
    
    path = './physionet_challenge/training2017/'
    
    list_classes = []
    thesaurus = {'N':'Normal','A':'AF','O':'Other Rhyth','~':'Noisy'}
    
    for fname in list(glob.glob(path+'*.mat')):
        ecg,header = read_challenge_mat_files(os.path.basename(fname),path)
        if header['siglen'] < 9000:
            class_ecg = get_class(header)
            list_classes.append(thesaurus[class_ecg])

    classes_less9000 = count(list_classes)
   
    return classes_less9000
    

def filter_data():
    
    ecgs = []
    tags = []
    path = './physionet_challenge/training2017/'
    thesaurus = {'N':'Normal','A':'AF','O':'Other Rhyth','~':'Noisy'}
    
    for fname in list(glob.glob(path+'*.mat')):
        ecg,header = read_challenge_mat_files(os.path.basename(fname),path)
        if header['siglen'] == 9000:
            ecgs.append(ecg)
            class_ecg = get_class(header)
            tags.append(thesaurus[class_ecg])
        
    data_ecgs = np.asarray(ecgs)
    data_tags = np.asarray(tags)
   
    return data_ecgs, data_tags

#def divide_data():
    
    #np.set_printoptions(threshold='nan')
    #ecgs,tags = filter_data()
    #skf = StratifiedShuffleSplit(n_splits = 1,test_size = 0.20)
    
    #return skf,ecgs,tags
    #for train, test in skf.split(ecgs,tags):
        #print("%s\n  \n%s\n" % (train,test))

#def balanced_data():

    #X,y = filter_data()  
    #X_res, y_res = SMOTE(kind='svm').fit_sample(X, y)
    #X_res, y_res = SMOTE().fit_sample(X, y)
    #y_count = count(y_res)
   
    #return y_count
    
    
      
#if __name__ == "__main__":
    
    #print get_distribution_classes()
    #print get_distribution_length()
    #print get_distribution_less9000()
    
   
#to modify
"""
def deep_conv_lstm_net():

    model = Sequential()
    model.add(Reshape(input_shape = (dim_length,dim_channels), target_shape =(dim_length,dim_channels,1)

    for i in range(9):
        model.add(Convolution2D(64,kernel_size=(3,1),padding='same',kernel_regularizer = 12(0.01)))
        model.add(Activation('relu'))

    model.add(Reshape(target_shape= (dim_length,64*dim_channels)))

    for i in range(3):
        model.add(LSTM(units=64,return_sequences=True,activation = 'tanh'))

    model.add(Dropout(0.5))
    model.add(TimeDistributed(Dense(units=output_dim)))
    model.add(Activation("softmax"))
    model.add(Lambda(lambda x: x[:,-1,:], output_shape[output_dim]))
    model.compile(loss='categorical_crossentropy',optimizer =Adam(lr=0.001), metrics = ['accuracy'])
"""

#plot_all_records(plot_original=False)