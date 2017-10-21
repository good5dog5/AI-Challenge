import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder

# util
def show_train_history(train_history,train,validation):
    import matplotlib.pyplot as plt
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train','validation'], loc='upper left')
    plt.show()

# function for read data
def group_encoder(vector) :
    onehot_encoder = OneHotEncoder(sparse=False) # the ans should be a dense matrix
    reshape_vector = vector.reshape(len(vector), 1)
    one_hot_matrix = onehot_encoder.fit_transform(reshape_vector)
    return  one_hot_matrix

def matrix_normalize(mat) :
    ans = mat.copy()
    shape = mat.shape
    for ii in range(0, shape[1], 1) :
        ans[:,ii] = preprocessing.scale(mat[:,ii])

    return ans

def readfile(path, date) :
    import pandas as pd
    dir_name_train  = "./stock_train_data_"
    dir_name_test   = "./stock_test_data_"
    full_path_train = path + dir_name_train + date + ".csv"
    full_path_test  = path + dir_name_test  + date + ".csv"
    raw_train_data  = pd.read_csv(full_path_train, sep=',', delimiter=None)
    raw_test_data   = pd.read_csv(full_path_test, sep=',', delimiter=None)
    cpy_train_data  = raw_train_data.copy(deep = True)
    cpy_test_data   = raw_test_data.copy(deep=True)

    # padding data here
    # ... ... ...

    # normalize weight
    cpy_train_data['weight'] = preprocessing.scale(raw_train_data['weight'])
    weightmean = raw_train_data['weight'].mean()
    cpy_train_data['weight'] = cpy_train_data['weight'] / weightmean

    return cpy_train_data, cpy_test_data

# kNN algorithm
def load_kNN(lib_name) :
    import ctypes
    c_ptr_obj_int    = np.ctypeslib.ndpointer(dtype=np.int32,   ndim=1, flags='C_CONTIGUOUS')
    c_ptr_obj_float  = np.ctypeslib.ndpointer(dtype=np.float32, ndim=1, flags='C_CONTIGUOUS')
    c_ptr_obj_double = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')
    libcfunc = ctypes.cdll.LoadLibrary(lib_name)
    libcfunc.kernel.restype = None
    libcfunc.kernel.argtypes = (c_ptr_obj_int, ctypes.c_int,
                                c_ptr_obj_float,
                                c_ptr_obj_int,
                                ctypes.c_int, ctypes.c_int)
    return libcfunc

def run_kNN(libkNN, k, weight, group, NumData, Numfeature) :
    weight_1d = np.array(weight).flatten()
    group__1d = np.array(group).flatten()
    result_1d = np.zeros(NumData * k)

    weight_1d = weight_1d.astype(np.float32)
    group__1d = group__1d.astype(np.int32)
    result_1d = result_1d.astype(np.int32)

    libkNN.kernel(result_1d, k, weight_1d, group__1d, NumData, Numfeature)
    result = np.reshape(result_1d,(NumData,k))
    return result

def encoder_result_kNN(group) :
    result_int = np.zeros(group.shape[0])
    for ii in range(0, group.shape[1], 1):
        result_int = result_int + (2 ** ii) * group[:,ii]

    result = group_encoder(result_int)
    return result

# build model
def build_nn_model(k) :
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Activation
    model = Sequential()
    model.add(Dense(units=1, input_dim=k, kernel_initializer='normal', activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def print_prob(model):
    for layer in model.layers:
        g=layer.get_config()
        h=layer.get_weights()
        print (g)
        print (h)


#### read and process the data ###
# read raw data
train_data, test_data = readfile('./', '20170910')

# onehot encoder
one_hot_group_train = group_encoder(train_data['group'].as_matrix())
one_hot_group_test  = group_encoder(test_data['group'].as_matrix())

# generate features and group into a large matrix
comp_mat_train = np.concatenate((train_data.iloc[:,1:89],one_hot_group_train),axis=1)
comp_mat_test  = np.concatenate((test_data.iloc [:,1:89],one_hot_group_test),axis=1)

# normalize
comp_mat_train = matrix_normalize(comp_mat_train)
comp_mat_test  = matrix_normalize(comp_mat_test)

# save matrix
np.save('comp_mat_train', comp_mat_train)
np.save('comp_mat_test' , comp_mat_test)

# combine train and test
#kNN_W = np.concatenate((comp_mat_train, comp_mat_test),axis=0)
#kNN_L = np.concatenate((train_data['label'].as_matrix(), test_data['label'].as_matrix()),axis=0)
kNN_W = comp_mat_train
kNN_L = train_data['label'].as_matrix()

#### read and process the data ###
print("run kNN...")
Sample = 5000
#Sample = train_data.shape[0]
k = 1
libc_kNN = load_kNN('./libkernel.so')
result = run_kNN(libc_kNN, k, kNN_W, kNN_L,Sample,kNN_W.shape[1])
result_encode = encoder_result_kNN(result)

### train probability ###
print("run training...")
SizeTrain = train_data.shape[0]
SizeTest  = test_data.shape[0]
train_W = train_data['weight'].as_matrix()
model = build_nn_model(2**k)
train_history=model.fit(x=result_encode[0:SizeTrain,:],
                        y=kNN_L[:SizeTrain],
                        sample_weight=train_W[0:SizeTrain],
                        validation_split=0.1,
                        epochs=100, batch_size=200,
                        verbose=2)
# get probability
prob = model.layers[0].get_weights()[0]
print_prob(model)

# predict part ... ... 

print("finish...")