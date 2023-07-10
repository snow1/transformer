import numpy as np
import scipy.io
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

# Load the EEG data and labels
total_data = scipy.io.loadmat('data/mat/A01T.mat')
train_data = total_data['data']
train_label = total_data['label']

train_data = np.transpose(train_data, (2, 1, 0))
train_data = np.expand_dims(train_data, axis=1)
train_label = np.transpose(train_label)

allData = train_data
allLabel = train_label[0]

# Load the test data
test_tmp = scipy.io.loadmat('data/mat/A04E.mat')
test_data = test_tmp['data']
test_label = test_tmp['label']

test_data = np.transpose(test_data, (2, 1, 0))
test_data = np.expand_dims(test_data, axis=1)
test_label = np.transpose(test_label)

testData = test_data
testLabel = test_label[0]

# Standardize the data
target_mean = np.mean(allData)
target_std = np.std(allData)
allData = (allData - target_mean) / target_std
testData = (testData - target_mean) / target_std

# Reshape the data for classification
data_train = np.reshape(allData, (allData.shape[0], -1))
data_test = np.reshape(testData, (testData.shape[0], -1))

# Train a classifier on the data
classifier = SVC(kernel='linear')
classifier.fit(data_train, allLabel)

# Predict labels for the testing data
predictions = classifier.predict(data_test)

# Calculate accuracy
accuracy = accuracy_score(testLabel, predictions)
print('Accuracy: {:.2f}%'.format(accuracy * 100))
