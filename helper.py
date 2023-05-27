import mne.io
import numpy as np


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