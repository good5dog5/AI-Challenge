#python-how-to-import-other-python-files
#https://stackoverflow.com/questions/2349991/python-how-to-import-other-python-files

#python tab/space
#http://blog.csdn.net/u012996583/article/details/36896705

import os
import pandas as pd
import pandas
import numpy as np
import matplotlib.pyplot as plt
import pylab
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from sklearn.preprocessing import normalize
from sklearn import preprocessing
from keras.utils import np_utils

def numpy_test():
	#a=np.array([0,1,2,3])
	arr = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
	np.delete(arr, np.s_[::2], 1)
	print(np.shape)

def Initial_train_test(Ini_datapath, date, broken_feature):
    from os import path
    #Read Initial Data
    trainDataPath = path.join(os.path.abspath(Ini_datapath), 'stock_train_data_2017'+date+'.csv')
    testDataPath = path.join(os.path.abspath(Ini_datapath), 'stock_test_data_2017'+date+'.csv')
    train_ini_00 = pd.read_csv(trainDataPath, sep=',', delimiter=None)
    test_ini_00 = pd.read_csv(testDataPath, sep=',', delimiter=None)

    #Deal with Broken Feature (這幾期主辦單位給的feature有少)
    temp_str='feature'+str(broken_feature)
    train_ini_00[temp_str]=np.zeros(len(train_ini_00))
    test_ini_00[temp_str]=np.zeros(len(test_ini_00))

    #normalize feature
    train_ini=train_ini_00.copy(deep=True)
    test_ini=test_ini_00.copy(deep=True)
    for i in range(1,89,1):
        train_ini.iloc[:,i]=preprocessing.scale(train_ini.iloc[:,i])
        test_ini.iloc[:,i]=preprocessing.scale(test_ini.iloc[:,i]) 

        #shift group number [1,28]->[0,27]
        test_ini['group']=test_ini['group']-1
        train_ini['group']=train_ini['group']-1

        #DataFrame to matrix
        train_array_ini=pd.DataFrame.as_matrix(train_ini)
        test_array_ini=pd.DataFrame.as_matrix(test_ini)

        ##remove useless column
        train_array=train_array_ini.copy()
        test_array=test_array_ini.copy()
        train_array=np.delete(train_array, np.s_[89:93], 1)
        train_array=np.delete(train_array, 0, 1)
        test_array=np.delete(test_array, 89, 1)
        test_array=np.delete(test_array, 0, 1)

        #Fill Broken Feature with ID
        for i in range(0,train_array.shape[0],1):
            train_array[i,43]=i
        for i in range(0,test_array.shape[0],1):
            test_array[i,43]=i+test_array.shape[0]    
        CAT_Array=np.concatenate((train_array, test_array), axis=0)
        CAT_Array=preprocessing.scale(CAT_Array)

        ##onehot encode (group)
        group_test_OneHot = np_utils.to_categorical(test_ini['group'], num_classes=max(test_ini['group'])+1)
        group_train_OneHot = np_utils.to_categorical(train_ini['group'], num_classes=max(train_ini['group'])+1)	

        CAT_GroupOneHot=np.concatenate((group_train_OneHot, group_test_OneHot), axis=0)
        CAT_GroupOneHot=preprocessing.scale(CAT_GroupOneHot)
        #XTrain=CAT[0:train_array1.shape[0],:]
        #XTest=CAT[train_array1.shape[0]:train_array1.shape[0]+test_array1.shape[0],:]

        ##combine array and onehot
        CAT=np.concatenate((CAT_Array, CAT_GroupOneHot), axis=1)
        XTrain=CAT[0:train_array1.shape[0],:]
        XTest=CAT[train_array1.shape[0]:train_array1.shape[0]+test_array1.shape[0],:]

        ##save preprocessing-data
        np.save('Xtrain', XTrain)
        #np.save('Xtrain', train_array_ini)
        np.save('Xtest', XTest)
        #np.save('Xtest', test_array_ini)
        np.save('Ytrain',pd.DataFrame.as_matrix(train_ini_00['label']))

def hi():
	print("hello/n")
	
def show_train_history(train_history,train,validation):
	#plot train history(with accuracy,loss)
	plt.plot(train_history.history[train])
	plt.plot(train_history.history[validation])
	plt.title('Train History')
	plt.ylabel(train)
	plt.xlabel('Epoch')
	plt.legend(['train','validation'], loc='upper left')
	plt.show()	
			
	#Show Image
	from PIL import Image
	file = Image.open("img.png")

def my_onehot(t):
	from numpy import array
	from numpy import argmax
	from keras.utils import to_categorical
	import pandas as pd
	# define example
	data = t
	data = array(data)
	#print(data)
	# one hot encode
	encoded = to_categorical(data)
	#print(encoded)
	# invert encoding
	inverted = argmax(encoded[0])
	#print(inverted)
	return encoded
    
def KNN_K1(Xtrain,Ytrian,Xtest,err,Name):
	from sklearn.neighbors.nearest_centroid import NearestCentroid
	clf = NearestCentroid()
	clf.fit(Xtrain, Ytrian)
	a=clf.predict(Xtest)
	result = pd.read_csv("./upload"+".csv", sep=',', delimiter=None)
	
	result['proba']=a*(1.0-err[0])+(1-a)*err[1]
	result.to_csv(Name[0]+str(Name[1])+".csv", sep=',', encoding='utf-8', index=False)
