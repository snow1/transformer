import mne.io
import numpy as np
import scipy

'''
print('check npy file')
data = np.load('C:/Users/Snow/Desktop/EEG/EEG_Networks-main/EEG_Networks-main/GAN/generate_label_subject1.npy')
print(data.shape)#(800, 4)
print(data[0])#[0 0 1 0]
print(data[1])

data2 = np.load('C:/Users/Snow/Desktop/EEG/EEG_Networks-main/EEG_Networks-main/GAN/WGAN_generate_X_subject1.npy')
print(data2.shape)#(800, 22, 250, 1)
print(data2[0])
'''
#mat file change to X_test.npy
#all 9 subject file into one file
def generate_data():
    for i in range(1, 10):
        mat = scipy.io.loadmat("C:/Users/Snow/Desktop/EEG/BCICIV_2a_gdf/standard_2a_data/A0%dE.mat" % i)
        data = mat['data'] 
        labels = mat['label']
        print(data.shape) #(1000, 22,288)
        print(labels.shape) #(288, 1)
        # data 288 swap to the first dimension
        data = np.swapaxes(data, 0, 2)
        print(data.shape) #(288, 22, 1000)
        # combine 9 subject data into one file
        if i == 1:
            concat_data = data
            concat_labels = labels
        else:
            concat_data = np.concatenate((concat_data, data), axis=0)
            concat_labels = np.concatenate((concat_labels, labels), axis=0)
    print(concat_data.shape) #(2592, 22, 1000)
    print(concat_labels.shape) #(2592, 1)
    np.save("C:/Users/Snow/Desktop/EEG/transformer/data/true_data/y_test.npy", concat_labels)
    np.save("C:/Users/Snow/Desktop/EEG/transformer/data/true_data/X_test.npy", concat_data)
    # np.save("C:/Users/Snow/Desktop/EEG/transformer/data/true_data/y_train_valid.npy", labels)#(576, 1)
    # np.save("C:/Users/Snow/Desktop/EEG/transformer/data/true_data/X_train_valid.npy", data)#(2000, 22, 288)

#generate_data()
mat = scipy.io.loadmat("C:/Users/Snow/Desktop/EEG/BCICIV_2a_gdf/standard_2a_data/A01E.mat" )
data = mat['data'] 
labels = mat['label']
print(data.shape) #(1000, 22,288)
print(labels.shape) #(288, 1)
# data 288 swap to the first dimension
data = np.swapaxes(data, 0, 2)
print(data.shape) #(288, 22, 1000)
np.save("C:/Users/Snow/Desktop/EEG/transformer/data/true_data/y_test.npy", labels)
np.save("C:/Users/Snow/Desktop/EEG/transformer/data/true_data/X_test.npy", data)

mat = scipy.io.loadmat("C:/Users/Snow/Desktop/EEG/BCICIV_2a_gdf/standard_2a_data/A01T.mat" )
data = mat['data'] 
labels = mat['label']
print(data.shape) #(1000, 22,288)
print(labels.shape) #(288, 1)
# data 288 swap to the first dimension
data = np.swapaxes(data, 0, 2)
print(data.shape) #(288, 22, 1000)
np.save("C:/Users/Snow/Desktop/EEG/transformer/data/true_data/y_train_valid.npy", labels)
np.save("C:/Users/Snow/Desktop/EEG/transformer/data/true_data/x_train_valid.npy", data)

'''
#combine all the npy file into one
for i in range(1, 10):
    data = np.load("C:/Users/Snow/Desktop/EEG/transformer/data/test_data_%d.npy" % i)
    print(data.shape)
    if i == 1:
        concat_data = data
    else:
        concat_data = np.concatenate((concat_data, data), axis=0)

np.save("C:/Users/Snow/Desktop/EEG/transformer/data/test_data.npy", concat_data)
print(concat_data.shape)
'''
# read the raw data
# for i in range(1, 10):
#     edf_raw = mne.io.read_raw_gdf("C:/Users/Snow/Desktop/EEG/BCICIV_2a_gdf/A0%dT.gdf" % i)
#     print(edf_raw.info)
#     edf_data, times = edf_raw[:, :]
#     np.savez_compressed("C:/Users/Snow/Desktop/EEG/BCICIV_2a_gdf/A0%dT" % i, data=edf_data, labels=i)

#edf_raw = mne.io.read_raw_gdf("C:/Users/Snow/Desktop/EEG/BCICIV_2a_gdf/A01E.gdf")
#hz = 1 / (edf_raw.times[1] - edf_raw.times[0])

# If you wish to get specific channels and time:
#edf_data, times = edf_raw[channels_indices, int(from_t * hz): int(to_t * hz]

# Or to get all the data:
#edf_data, times = edf_raw[:, :]

# save the raw data in ny format and add labels
#np.save("C:/Users/Snow/Desktop/EEG/BCICIV_2a_gdf/A01E.npy", edf_data)
#np.savez_compressed("C:/Users/Snow/Desktop/EEG/BCICIV_2a_gdf/A01E", data=edf_data, labels=1)
# load npz file and check the labels
#data = np.load("C:/Users/Snow/Desktop/EEG/BCICIV_2a_gdf/A01E.npz")
#print(data["labels"])