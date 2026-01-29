import numpy as np
import pandas as pd
import h5py

#Raw data csv files 
kaykay_raw1 = pd.read_csv('walking_kaykay.csv')
kaykay_raw2 = pd.read_csv('jumping_kaykay.csv')
lorenzo_raw1 = pd.read_csv('walking_lorenzo.csv')
lorenzo_raw2 = pd.read_csv('jumping_lorenzo.csv')
# daniil_raw1=pd.read_csv()
# daniil_raw2=pd.read_csv()

#pre-processed data csv files
kaykay_pp1=pd.read_csv('jumping_kaykay_processed.csv')
kaykay_pp2=pd.read_csv('walking_kaykay_processed.csv')
lorenzo_pp1=pd.read_csv('jumping_lorenzo_processed.csv')
lorenzo_pp2=pd.read_csv('walking_lorenzo_processed.csv')
# daniil_pp1=pd.read_csv()
# daniil_pp2=pd.read_csv()

with h5py.File('data_structure.h5', 'w') as hdf:
    #create raw group data
    G1 = hdf.create_group('Raw Data')

    #create sub groups for each person
    G1_kaykay = G1.create_group('KayKay Raw Data')
    G1_lorenzo = G1.create_group('Lorenzo Raw Data')
    # G1_daniil = G1.create_group('Daniil Raw Data')

    #respective walking and jumping fata sets
    G1_kaykay.create_dataset('KayKay Walking Raw', data=kaykay_raw1.values)
    G1_kaykay.create_dataset('KayKay Jumping Raw', data=kaykay_raw2.values)

    G1_lorenzo.create_dataset('Lorenzo Walking Raw', data=lorenzo_raw1.values)
    G1_lorenzo.create_dataset('Lorenzo Jumping Raw', data=lorenzo_raw2.values)

    # G1_daniil.create_dataset('Daniil Walking Raw', data=daniil_raw1.values)
    # G1_daniil.create_dataset('Daniil Jumping Raw', data=daniil_raw2.values)

    #create pre-processed data
    G2=hdf.create_group('Pre-processed Data')

    #create sub groups for each person
    G2_kaykay = G2.create_group('KayKay Pre-processed Data')
    G2_lorenzo = G2.create_group('Lorenzo Pre-processed Data')
    # G2_daniil = G2.create_group('Daniil Pre-processed Data')
    #
    G1_kaykay.create_dataset('KayKay Walking Pre-processed', data=kaykay_pp1.values)
    G1_kaykay.create_dataset('KayKay Jumping Pre-processed', data=kaykay_pp2.values)
    #
    G1_lorenzo.create_dataset('Lorenzo Walking Pre-processed', data=lorenzo_pp1.values)
    G1_lorenzo.create_dataset('Lorenzo Jumping Pre-processed', data=lorenzo_pp2.values)

    # G1_daniil.create_dataset('Daniil Walking Pre-processed', data=)
    # G1_daniil.create_dataset('Daniil Jumping Pre-processed', data=)

    #create segmented data group
    # G3=hdf.create_group('Segmented Data')

    #create subgroups for train and test
    # G3_train=G3.create_group('Train')
    # G3_test=G3.create_group('Test')

    #create data sets for segmented data
    # G3_train.create_dataset('Train Data', data=)
    # G3_test.create_dataset('Test Data', data=)
