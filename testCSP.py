import numpy as np
from numpy.linalg import eig
import scipy.io
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from common_spatial_pattern import csp1
from mne.decoding import CSP

# Load the EEG data and labels
        # to get the data of target subject
total_data = scipy.io.loadmat('data/mat/A04T.mat')
train_data = total_data['data']
train_label = total_data['label']
        

train_data = np.transpose(train_data, (2, 1, 0))
train_data = np.expand_dims(train_data, axis=1)
train_label = np.transpose(train_label)

allData = train_data
allLabel = train_label[0]


 # test data
        # to get the data of target subject
test_tmp = scipy.io.loadmat('data/mat/A04E.mat')
test_data = test_tmp['data']
test_label = test_tmp['label']

        # self.train_data = self.train_data[250:1000, :, :]
test_data = np.transpose(test_data, (2, 1, 0))
test_data = np.expand_dims(test_data, axis=1)
test_label = np.transpose(test_label)

testData = test_data
testLabel = test_label[0]

        # standardize
target_mean = np.mean(allData)
target_std = np.std(allData)
allData = (allData - target_mean) / target_std
testData = (testData - target_mean) / target_std

tmp_alldata = np.transpose(np.squeeze(allData), (0, 2, 1))


Wb = csp1(tmp_alldata, allLabel-1)
#csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
#csp.fit_transform(tmp_alldata, allLabel-1)
#csp.plot_patterns(tmp_alldata, ch_type='eeg', units='Patterns (AU)', size=1.5)
#csp.plot_filters(tmp_alldata, ch_type='eeg', units='Patterns (AU)', size=1.5)
#show me the difference using wb or not

# Apply the CSP filters to the training and testing data
data_train_csp = np.matmul(Wb.T, allData)
data_test_csp = np.matmul(Wb.T, testData)

# Reshape the data for classification
data_train_csp = np.reshape(data_train_csp, (data_train_csp.shape[0], -1))
data_test_csp = np.reshape(data_test_csp, (data_test_csp.shape[0], -1))

# Train a classifier on the CSP-transformed data
classifier = SVC(kernel='linear')
classifier.fit(data_train_csp, (train_label-1).ravel())

# Predict labels for the testing data
predictions = classifier.predict(data_test_csp)

# Calculate accuracy
accuracy = accuracy_score((test_label-1).ravel(), predictions)

print('Accuracy: {:.2f}%'.format(accuracy * 100))
