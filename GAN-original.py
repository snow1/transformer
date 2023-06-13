#Load EEG-GAN module
from eeggan import train_gan, visualize_gan, generate_samples, setup_tutorial

#Load other modules specific to this notebook
import numpy as np
import matplotlib.pyplot as plt
import shutil
import os
import random as rnd
from scipy import signal
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import torch

#Create a print formatting class
class printFormat:
    bold = '\033[1m'
    italic = '\033[3m'
    end = '\033[0m'

#Setup 
#This function downloads tutorial-required files (e.g., datasets) from the GitHub. These files can also be found within the package itself, but Google Colab has difficulty accessing it.
#This function is only necessary when running the tutorial but it also creates three folders (data, trained_models, generated_samples) that are needed for the package, so still may be useful with your own data.
setup_tutorial()



################Load the data
empiricalHeaders = np.genfromtxt('data/csv/gansEEGTrainingData.csv', delimiter=',', names=True).dtype.names
empiricalEEG = np.genfromtxt('data/csv/gansEEGTrainingData.csv', delimiter=',', skip_header=1)

#Print the head of the data
print(printFormat.bold + 'Display Header and first few rows/columns of data\n \033[0m' + printFormat.end)
print(empiricalHeaders[:6])
print(empiricalEEG[0:3,:6])

#Print some information about the columns
print('\n------------------------------------------------------------------------------------------')
print(printFormat.bold + '\nNote the first three columns:' + printFormat.end +'\n    ParticipantID - Indicates different participants\n    Condition - Indicates the condition (WIN = 0, LOSE = 1) to be classified\n    Trial - Indicates the trial number for that participant and condition')
print('\nThe remaining columns are titled Time1 to Time100 - indicating 100 datapoints per sample.\nThe samples span from -200 to 1000ms around the onset of a feedback stimulus.\nThese are downsampled from the original data, which contained 600 datapoints per sample.')

#Print some meta-data
print('\n------------------------------------------------------------------------------------------')
print('\n' + printFormat.bold + 'Other characteristics of our data include:' + printFormat.end)
print('-We have ' + str(len(set(empiricalEEG[:,0]))) + ' participants in our training set')
print('-Participants have an average of ' + str(round(np.mean([np.max(empiricalEEG[empiricalEEG[:,0]==pID,2]) for pID in set(empiricalEEG[:,0])]))) + ' (SD: ' + str(round(np.std([np.max(empiricalEEG[empiricalEEG[:,0]==pID,2]) for pID in set(empiricalEEG[:,0])]))) + ')' + ' trials per outcome (win, lose)')


################view the data
#Determine which rows are each condition
lossIndex = np.where(empiricalEEG[:,1]==1)
winIndex = np.where(empiricalEEG[:,1]==0)

#Grand average the waveforms for each condition
lossWaveform = np.mean(empiricalEEG[lossIndex,3:],axis=1)[0]
winWaveform = np.mean(empiricalEEG[winIndex,3:],axis=1)[0]

#Determine x axis of time
time = np.linspace(-200,1000,100)

#Setup figure
f, (ax1) = plt.subplots(1, 1, figsize=(6, 4))

#Plot each waveform
ax1.plot(time, lossWaveform, label = 'Loss')
ax1.plot(time, winWaveform, label = 'Win')

#Format plot
ax1.set_ylabel('Voltage ($\mu$V)')
ax1.set_xlabel('Time (ms)')
ax1.set_title('Empirical', loc='left')
ax1.spines[['right', 'top']].set_visible(False)
ax1.legend(frameon=False)

###################train the GAN
argv = dict(
    path_dataset="data/csv/gansEEGTrainingData.csv",
    n_epochs=5
)

train_gan(argv)

#Train the GAN on GPUs
#Note, on Google Colab you can start a GPU runtime by going to Runtime > Change runtime type > Hardware accelerator > GPU
'''
argv = dict(
    ddp = True,
    path_dataset="data/gansEEGTrainingData.csv",
    n_epochs=5
)

train_gan(argv)
'''

###########Visualize the GAN
#We trained our GAN for 5 epochs and this will result in a model that is severally under-trained, so we will instead use a pre-trained GAN that trained for 8000 epochs:
argv = dict(
    plot_losses = True,
    checkpoint = True,
    file = "gansEEGModel.pt",
    training_file = "data\csv\gansEEGTrainingData.csv"
)

visualize_gan(argv)

#The GAN training results fom the last step results in a file named checkpoint.pt. If you want to continue with this file, use the following line of code:
'''
argv = dict(
    plot_losses = True,
    checkpoint = True,
    file = "checkpoint.pt",
    training_file = "data\gansEEGTrainingData.csv"
)

visualize_gan(argv)
'''
#################Generate synthetic data
#We trained our GAN for 5 epochs and this will result in a model that is severally under-trained, so we will instead use a pre-trained GAN that trained for 8000 epochs:
argv = dict(
    file = "gansEEGModel.pt",
    path_samples = "gansEEGSyntheticData.csv",
    num_samples_total = 10000
)

generate_samples(argv)

#The GAN training results fom the last step results in a file named checkpoint.pt. If you want to continue with this file, use the following line of code:
'''
argv = dict(
    file = "checkpoint.pt",
    path_samples = "gansEEGSyntheticData.csv",
    num_samples_total = 10000
)

generate_samples(argv)
'''


###############Load Data 
syntheticEEG = np.genfromtxt('generated_samples/gansEEGSyntheticData.csv', delimiter=',', skip_header=1)

#Print head of the data
print(printFormat.bold + 'Display first few rows/columns of data' + printFormat.end)
print(['Condition','Time1','Time2','Time3','Time4','Time5'])
print(syntheticEEG[0:3,0:6])

#Print condition sample counts
print('\n' + printFormat.bold + 'Display trial counts for each condition' + printFormat.end)
print(printFormat.bold +'Win: ' + printFormat.end + str(np.sum(syntheticEEG[:,0]==0)))
print(printFormat.bold +'Lose: ' + printFormat.end + str(np.sum(syntheticEEG[:,0]==1)))



###############View the data
#Determine 5 random trials to plot
empiricalIndex = rnd.sample(range(0, empiricalEEG.shape[0]), 5)
syntheticIndex = rnd.sample(range(0, syntheticEEG.shape[0]), 5)

#Plot trial data
f, ax = plt.subplots(5, 2, figsize=(12, 4))
for c in range(5):
    ax[c,0].plot(time,empiricalEEG[empiricalIndex[c],3:]) #Note, we here add the same filter simply for visualization
    ax[c,0].set_yticks([])
    
    ax[c,1].plot(time,syntheticEEG[syntheticIndex[c],1:])
    ax[c,1].spines[['left', 'right', 'top']].set_visible(False)
    ax[c,1].set_yticks([])
    
    if c == 0:
        ax[c,0].set_title('Empirical', loc='left')
        ax[c,1].set_title('Synthetic', loc='left')
    else:
        ax[c,0].set_title(' ')
        ax[c,1].set_title(' ')
        
    if c != 4:
        ax[c,0].spines[['bottom', 'left', 'right', 'top']].set_visible(False)
        ax[c,1].spines[['bottom', 'left', 'right', 'top']].set_visible(False)
        ax[c,0].set_xticks([])
        ax[c,1].set_xticks([])
    else:
        ax[c,0].spines[['left', 'right', 'top']].set_visible(False)
        ax[c,1].spines[['left', 'right', 'top']].set_visible(False)
        ax[c,0].set_xlabel('Time (ms)')
        ax[c,1].set_xlabel('Time (ms)')



###############View ERP Data
#Grand average the synthetic waveforms for each condition
synLossWaveform = np.mean(syntheticEEG[np.r_[syntheticEEG[:,0]==1],1:],axis=0)
synWinWaveform = np.mean(syntheticEEG[np.r_[syntheticEEG[:,0]==0],1:],axis=0)

#Set up figure
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

#Plot each empirical waveform (note, we here add the same processing simply for visualization)
ax1.plot(time, scale(winWaveform), label = 'Empirical')
ax1.plot(time, scale(synWinWaveform), label = 'Synthetic')

#Format plot
ax1.set_ylabel('Voltage (μμ\muV)')
ax1.set_xlabel('Time (ms)')
ax1.set_title('Win', loc='left')
ax1.spines[['right', 'top']].set_visible(False)
ax1.tick_params(left = False, labelleft = False)
ax1.legend(frameon=False)

#Plot each synthetic waveform
ax2.plot(time, scale(lossWaveform), label = 'Empirical')
ax2.plot(time, scale(synLossWaveform), label = 'Synthetic')

#Format plot
ax2.set_ylabel('Voltage (μμ\muV)')
ax2.set_xlabel('Time (ms)')
ax2.set_title('Lose', loc='left')
ax2.spines[['right', 'top']].set_visible(False)
ax2.tick_params(left = False, labelleft = False)
ax2.legend(frameon=False)


# Step 4. Classification
# Step 4.1. Preparing Validation Data
#Set seed for a bit of reproducibility
rnd.seed(1618)

#This function averages trial-level empirical data for each participant and condition
def averageEEG(EEG):
    participants = np.unique(EEG[:,0])
    averagedEEG = []
    for participant in participants:
        for condition in range(2):
            averagedEEG.append(np.mean(EEG[(EEG[:,0]==participant)&(EEG[:,1]==condition),:], axis=0))
    return np.array(averagedEEG)

#Load test data to predict (data that neither the GAN nor the classifier will ever see in training)
EEGDataTest = np.genfromtxt('data/csv/gansEEGValidationData.csv', delimiter=',', skip_header=1)
EEGDataTest = averageEEG(EEGDataTest)[:,1:]

#Extract test outcome and predictor data
y_test = EEGDataTest[:,0]
x_test = EEGDataTest[:,2:]
x_test = scale(x_test,axis = 1)


#############prepare Empirical data
#Create participant by condition averages
Emp_train = averageEEG(empiricalEEG)[:,1:]

#Extract the outcomes
Emp_Y_train = Emp_train[:,0]

#Scale the predictors
Emp_X_train = scale(Emp_train[:,2:], axis=1)

#Shuffle the order of samples
trainShuffle = rnd.sample(range(len(Emp_X_train)),len(Emp_X_train))
Emp_Y_train = Emp_Y_train[trainShuffle]
Emp_X_train = Emp_X_train[trainShuffle,:]

########### Preparing Augmented Data
#This function averages trial-level synthetic data in bundles of 50 trials, constrained to each condition
def averageSynthetic(synData):
    samplesToAverage = 50

    lossSynData = synData[synData[:,0]==0,:]
    winSynData = synData[synData[:,0]==1,:]

    lossTimeIndices = np.arange(0,lossSynData.shape[0],samplesToAverage)
    winTimeIndices = np.arange(0,winSynData.shape[0],samplesToAverage)
    
    newLossSynData = [np.insert(np.mean(lossSynData[int(trialIndex):int(trialIndex)+samplesToAverage,1:],axis=0),0,0) for trialIndex in lossTimeIndices]
    newWinSynData = [np.insert(np.mean(winSynData[int(trialIndex):int(trialIndex)+samplesToAverage,1:],axis=0),0,1) for trialIndex in winTimeIndices]

    avgSynData = np.vstack((np.asarray(newLossSynData),np.asarray(newWinSynData)))
    
    return avgSynData

#Create 'participant' by condition averages
Syn_train = averageSynthetic(syntheticEEG)

#Extract the outcomes
Syn_Y_train = Syn_train[:,0]

#Scale the predictors
Syn_X_train = scale(Syn_train[:,1:], axis=1)

#Combine empirical and synthetic datasets to create an augmented dataset
Aug_Y_train = np.concatenate((Emp_Y_train,Syn_Y_train))
Aug_X_train = np.concatenate((Emp_X_train,Syn_X_train))

#Shuffle the order of samples
trainShuffle = rnd.sample(range(len(Aug_X_train)),len(Aug_X_train))
Aug_Y_train = Aug_Y_train[trainShuffle]
Aug_X_train = Aug_X_train[trainShuffle,:]




#############Step 5. Support Vector Machine
#############Step 5.1. Define Search Space
#Determine SVM search space
param_grid_SVM = [
    {'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf', 'poly', 'sigmoid']}]

########## Classify Empirical Data
#Setup tracking variable
predictionScores_SVM = []

#Setup SVM grid search
optimal_params = GridSearchCV(
    SVC(), 
    param_grid_SVM, 
    refit = True, 
    verbose = False)

#Conduct classification
optimal_params.fit(Emp_X_train, Emp_Y_train)
SVMOutput = optimal_params.predict(x_test)

#Determine performance
predictResults = classification_report(y_test, SVMOutput, output_dict=True)
predictionScores_SVM.append(round(predictResults['accuracy']*100))





#############Step 5.3. Classify Augmented Data
#Setup SVM grid search
optimal_params = GridSearchCV(
    SVC(), 
    param_grid_SVM, 
    refit = True, 
    verbose = False)

#Conduct classification
optimal_params.fit(Aug_X_train, Aug_Y_train)
SVMOutput = optimal_params.predict(x_test)

#Determine performance
predictResults = classification_report(y_test, SVMOutput, output_dict=True)
predictionScores_SVM.append(round(predictResults['accuracy']*100))

#Report results
print('Empirical Classification Accuracy: ' + str(predictionScores_SVM[0]) + '%')
print('Augmented Classification Accuracy: ' + str(predictionScores_SVM[1]) + '%')




# Step 6. Neural Network
# Step 6.1. Define Search Space

#Determine neural network search space
param_grid_NN = [
    {'hidden_layer_sizes': [(25,), (50,), (25, 25), (50,50), (50,25,50)],
    'activation': ['logistic', 'tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant', 'invscaling', 'adaptive'],
    'max_iter' : [10000]}]


##########Step 6.2. Classify Empirical Data
#Signify computational time
print('This may take a few minutes...')

#Setup tracking variable
predictionScores_NN = []

#Setup neural network grid search
optimal_params = GridSearchCV(
    MLPClassifier(), 
    param_grid_NN, 
    verbose = True,
    n_jobs = -1)

#Conduct classification
optimal_params.fit(Emp_X_train, Emp_Y_train);
neuralNetOutput = MLPClassifier(hidden_layer_sizes=optimal_params.best_params_['hidden_layer_sizes'], 
                            activation=optimal_params.best_params_['activation'],
                            solver = optimal_params.best_params_['solver'], 
                            alpha = optimal_params.best_params_['alpha'], 
                            learning_rate = optimal_params.best_params_['learning_rate'], 
                            max_iter = optimal_params.best_params_['max_iter'])
neuralNetOutput.fit(Emp_X_train, Emp_Y_train)
y_true, y_pred = y_test , neuralNetOutput.predict(x_test)

#Determine performance
predictResults = classification_report(y_true, y_pred, output_dict=True)
predictScore = round(predictResults['accuracy']*100)
predictionScores_NN.append(predictScore)



##########Step 6.3. Classify Augmented Data
#Signify computational time
print('This may take twice as long as the empirical neural network classification...')

#Setup neural network grid search
optimal_params = GridSearchCV(
    MLPClassifier(), 
    param_grid_NN, 
    verbose = True,
    n_jobs = -1)

#Conduct classification
optimal_params.fit(Aug_X_train, Aug_Y_train);
neuralNetOutput = MLPClassifier(hidden_layer_sizes=optimal_params.best_params_['hidden_layer_sizes'], 
                            activation=optimal_params.best_params_['activation'],
                            solver = optimal_params.best_params_['solver'], 
                            alpha = optimal_params.best_params_['alpha'], 
                            learning_rate = optimal_params.best_params_['learning_rate'], 
                            max_iter = optimal_params.best_params_['max_iter'])
neuralNetOutput.fit(Aug_X_train, Aug_Y_train)
y_true, y_pred = y_test , neuralNetOutput.predict(x_test)

#Determine performance
predictResults = classification_report(y_true, y_pred, output_dict=True)
predictScore = round(predictResults['accuracy']*100)
predictionScores_NN.append(predictScore)

#Report results
print('Empirical Classification Accuracy: ' + str(predictionScores_NN[0]) + '%')
print('Augmented Classification Accuracy: ' + str(predictionScores_NN[1]) + '%')



#######Step 7. Final Report
########Step 7.1. Present Classification Performance
#Report results
print(printFormat.bold + 'SVM Classification Results:' + printFormat.end)
print('Empirical Classification Accuracy: ' + str(predictionScores_SVM[0]) + '%')
print('Augmented Classification Accuracy: ' + str(predictionScores_SVM[1]) + '%')

#Report results
print('\n' + printFormat.bold + 'Neural Network Classification Results:' + printFormat.end)
print('Empirical Classification Accuracy: ' + str(predictionScores_NN[0]) + '%')
print('Augmented Classification Accuracy: ' + str(predictionScores_NN[1]) + '%')
print('\n' + printFormat.italic + 'Note: Due to randomization in this process, these accuracies will vary.'+ printFormat.end)


#########Step 7.2. Plot Classification Performance¶
ax = plt.subplot(111) 
plt.bar([.9,1.9],[predictionScores_SVM[0],predictionScores_NN[0]], width=.2)
plt.bar([1.1,2.1],[predictionScores_SVM[1],predictionScores_NN[1]], width=.2)
plt.ylim([0,round((np.max([predictionScores_SVM,predictionScores_NN])+20)/10)*10])
predictionScores = predictionScores_SVM+predictionScores_NN
for xi, x in enumerate([.86,1.06,1.86,2.06]):
    plt.text(x,predictionScores[xi]+1,str(predictionScores[xi])+'%')
plt.xticks([1,2], labels = ['SVM', 'Neural Network'])
plt.legend(['Empirical','Augmented'], loc='upper right', frameon=False)
ax.spines[['right', 'top']].set_visible(False)