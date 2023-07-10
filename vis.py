# show the result of the model
import os
# import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
import re


def show_result():
    # CNN ; linearTest1 ; Example3 ; LSTMTest1 ; LSTMTes2 ;transformer222EMB_size5DEPTH2_0.Accuracy
    # transformer2Test; transformerEMB_size5DEPTH2 #transformerTestforMemory
    path = os.getcwd() + '/outputs/transformerTestforMemory/Markdown' 
    file = os.listdir(path) 
    testResult = []
    trainResult = []
    testLoss = []
    trainLoss = []
    #read the result
    for i in range(len(file)):
        if file[i].endswith('.md'):
            with open(path + '\\' + file[i], 'r') as f:
                result = f.read()
                result = result.split('\n')
                testResults = []
                trainResults = []
                trainLosss = []
                testLosss = []
                for j in range(len(result)):
                    if re.search('Test accuracy', result[j]):
                        result[j] = result[j].split(' ')
                        #print("result[j]",result[j])
                        #testResult = result[j][-1:][0]
                        testResult = result[j][-28:][0] # only for transformerEMB_size5DEPTH2
                        #print("testResult",testResult)
                        testResults.append(float(testResult))

                    if re.search('Train accuracy', str(result[j])):
                        #print("result[j]",result[j])
                        trainResult = result[j][-7:][0]
                        #print("trainResult",trainResult)
                        trainResults.append(float(trainResult))
                    # test loss
                    if re.search('Test loss', str(result[j])):
                        #print("result[j]",result[j])
                        testLoss = result[j][-12:][0]
                        #print("testLoss",testLoss)
                        testLosss.append(float(testLoss))

                    # train loss
                    if re.search('Train loss', str(result[j])):
                        #print("result[j]",result[j])
                        trainLoss = result[j][-17:][0]
                        #print("testLoss",testLoss)
                        trainLosss.append(float(trainLoss))

                # #plot train loss and test loss in one figure
                plt.plot(trainLosss, color='red', label='Train Loss')
                plt.plot(testLosss, color='blue', label='Test Loss')
                plt.ylim(0, 2)
                plt.xlim(0, 500)
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title('Train Loss and Test Loss')
                plt.legend()
                plt.savefig(path + '\\' + file[i][:-2] + 'Loss.png')
                plt.show()
                #save the result

                #plot the result
                plt.plot(trainResults, color='red', label='Train Accuracy')
                plt.plot(testResults, color='blue', label='Test Accuracy')
                plt.ylim(0, 1)
                plt.xlim(0, 500)
                plt.xlabel('Epoch')
                plt.ylabel('Accuracy')
                plt.title('Accuracy')
                plt.legend()
                plt.savefig(path + '\\' + file[i][:-2] + 'Accuracy.png')
                plt.show()
                #save the result
                plt.close()


#show_result()   
def show_result111():
    # CNN ; linearTest1 ; Example3 ; LSTMTest1 ; LSTMTes2 ;transformer222EMB_size5DEPTH2_0.Accuracy
    # transformer2Test; transformerEMB_size5DEPTH2 #transformerTestforMemory
    path = os.getcwd() 
    file = os.listdir(path) 
    testResult = []
    trainResult = []
    testLoss = []
    trainLoss = []
    #read the result
    for i in range(len(file)):
        if file[i].endswith('.txt'):
            with open(path + '\\' + file[i], 'r') as f:
                result = f.read()
                result = result.split('\n')
                testResults = []
                trainResults = []
                trainLosss = []
                testLosss = []
                for j in range(len(result)):
                    # if re.search('Test accuracy', result[j]):
                    #     result[j] = result[j].split(' ')
                    #     #print("result[j]",result[j])
                    #     #testResult = result[j][-1:][0]
                    #     testResult = result[j][-28:][0] # only for transformerEMB_size5DEPTH2
                    #     #print("testResult",testResult)
                    #     testResults.append(float(testResult))

                    # if re.search('Train accuracy', str(result[j])):
                    #     #print("result[j]",result[j])
                    #     trainResult = result[j][-7:][0]
                    #     #print("trainResult",trainResult)
                    #     trainResults.append(float(trainResult))
                    # test loss
                    if re.search('Test', result[j]):
                        #print("result[j]",result[j],len(result[j]))
                        result[j] = result[j].split(' ')
                        testLoss = result[j][-1:][0]
                        #delete the first and last character
                        testLoss = testLoss[1:-2]
                        #print("testLoss",testLoss)
                        #testLoss divided by 100
                        #``
                        testLosss.append(float(testLoss)/100)

                    # # train loss
                    # if re.search('Train loss', str(result[j])):
                    #     #print("result[j]",result[j])
                    #     trainLoss = result[j][-17:][0]
                    #     #print("testLoss",testLoss)
                    #     trainLosss.append(float(trainLoss))

                # #plot train loss and test loss in one figure
                #plt.plot(trainLosss, color='red', label='Train Loss')
                plt.plot(testLosss, color='blue', label='Test accuracy')
                plt.ylim(0, 0.8)
                plt.xlim(0, 100)
                plt.xlabel('Epoch')
                plt.ylabel('accuracy')
                plt.title('Test accuracy')
                plt.legend()
                plt.savefig(path + '\\' + file[i][:-2] + 'accuracy.png')
                plt.show()

#show_result111()
from keras.utils import plot_model
from keras.models import Sequential
from keras.layers import Conv2D, BatchNormalization, Dropout, Activation, AveragePooling2D, Flatten, Dense

def create_model():
    model = Sequential()
    model.add(Conv2D(filters=40,
                     kernel_size=(1, 25),
                     data_format='channels_last',
                     input_shape=(22, 534, 1)))
    model.add(BatchNormalization(axis=-1))
    model.add(Dropout(rate=0.5))
    model.add(Conv2D(filters=40,
                     data_format='channels_last',
                     kernel_size=(22, 1)))
    model.add(BatchNormalization(axis=-1))
    model.add(Activation('elu'))
    model.add(AveragePooling2D(pool_size=(1, 75), strides=(1, 15)))
    model.add(Activation('elu'))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(9, activation='softmax'))
    return model

#model = create_model()
#plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
import re
import matplotlib.pyplot as plt

def analyze_log(log):
    epochs = []
    loss_values = []
    accuracy_values = []

    pattern = r"Epoch (\d+)/\d+[\s\S]*?loss: ([\d.]+) - acc: ([\d.]+)"
    matches = re.findall(pattern, log)

    for match in matches:
        epoch = int(match[0])
        loss = float(match[1])
        accuracy = float(match[2])

        epochs.append(epoch)
        loss_values.append(loss)
        accuracy_values.append(accuracy)

    return epochs, loss_values, accuracy_values

# Example usage
log = """
Epoch 1/300
230/230 [==============================] - 5s 22ms/step - loss: 1.4706 - acc: 0.4348
Epoch 2/300
230/230 [==============================] - 0s 487us/step - loss: 1.2393 - acc: 0.6130
Epoch 3/300
230/230 [==============================] - 0s 448us/step - loss: 1.0366 - acc: 0.5391
Epoch 4/300
230/230 [==============================] - 0s 448us/step - loss: 0.9477 - acc: 0.6043
Epoch 5/300
230/230 [==============================] - 0s 578us/step - loss: 0.9174 - acc: 0.6000
Epoch 6/300
230/230 [==============================] - 0s 439us/step - loss: 0.8693 - acc: 0.6261
Epoch 7/300
230/230 [==============================] - 0s 478us/step - loss: 0.8751 - acc: 0.6087
Epoch 8/300
230/230 [==============================] - 0s 465us/step - loss: 0.8342 - acc: 0.6913
Epoch 9/300
230/230 [==============================] - 0s 470us/step - loss: 0.7957 - acc: 0.7391
Epoch 10/300
230/230 [==============================] - 0s 435us/step - loss: 0.7362 - acc: 0.7565
Epoch 11/300
230/230 [==============================] - 0s 457us/step - loss: 0.7614 - acc: 0.7174
Epoch 12/300
230/230 [==============================] - 0s 443us/step - loss: 0.6923 - acc: 0.7913
Epoch 13/300
230/230 [==============================] - 0s 457us/step - loss: 0.6691 - acc: 0.8130
Epoch 14/300
230/230 [==============================] - 0s 483us/step - loss: 0.6239 - acc: 0.7957
Epoch 15/300
230/230 [==============================] - 0s 461us/step - loss: 0.6183 - acc: 0.8348
Epoch 16/300
230/230 [==============================] - 0s 443us/step - loss: 0.6136 - acc: 0.8261
Epoch 17/300
230/230 [==============================] - 0s 444us/step - loss: 0.5818 - acc: 0.8391
Epoch 18/300
230/230 [==============================] - 0s 474us/step - loss: 0.5440 - acc: 0.8739
Epoch 19/300
230/230 [==============================] - 0s 439us/step - loss: 0.5946 - acc: 0.8174
Epoch 20/300
230/230 [==============================] - 0s 431us/step - loss: 0.5424 - acc: 0.8478
Epoch 21/300
230/230 [==============================] - 0s 422us/step - loss: 0.5065 - acc: 0.8870
Epoch 22/300
230/230 [==============================] - 0s 413us/step - loss: 0.4936 - acc: 0.8826
Epoch 23/300
230/230 [==============================] - 0s 448us/step - loss: 0.5023 - acc: 0.8870
Epoch 24/300
230/230 [==============================] - 0s 457us/step - loss: 0.5092 - acc: 0.8609
Epoch 25/300
230/230 [==============================] - 0s 439us/step - loss: 0.4992 - acc: 0.8652
Epoch 26/300
230/230 [==============================] - 0s 439us/step - loss: 0.4684 - acc: 0.8783
Epoch 27/300
230/230 [==============================] - 0s 439us/step - loss: 0.4793 - acc: 0.8783
Epoch 28/300
230/230 [==============================] - 0s 435us/step - loss: 0.4593 - acc: 0.8957
Epoch 29/300
230/230 [==============================] - 0s 465us/step - loss: 0.4344 - acc: 0.8957
Epoch 30/300
230/230 [==============================] - 0s 435us/step - loss: 0.4691 - acc: 0.8957
Epoch 31/300
230/230 [==============================] - 0s 404us/step - loss: 0.4669 - acc: 0.8826
Epoch 32/300
230/230 [==============================] - 0s 422us/step - loss: 0.4258 - acc: 0.9000
Epoch 33/300
230/230 [==============================] - 0s 457us/step - loss: 0.4218 - acc: 0.9043
Epoch 34/300
230/230 [==============================] - 0s 443us/step - loss: 0.4236 - acc: 0.9130
Epoch 35/300
230/230 [==============================] - 0s 413us/step - loss: 0.4182 - acc: 0.8826
Epoch 36/300
230/230 [==============================] - 0s 448us/step - loss: 0.4373 - acc: 0.8826
Epoch 37/300
230/230 [==============================] - 0s 430us/step - loss: 0.4367 - acc: 0.9130
Epoch 38/300
230/230 [==============================] - 0s 430us/step - loss: 0.4416 - acc: 0.9130
Epoch 39/300
230/230 [==============================] - 0s 422us/step - loss: 0.4303 - acc: 0.9130
Epoch 40/300
230/230 [==============================] - 0s 431us/step - loss: 0.4541 - acc: 0.9043
Epoch 41/300
230/230 [==============================] - 0s 426us/step - loss: 0.4113 - acc: 0.9043
Epoch 42/300
230/230 [==============================] - 0s 530us/step - loss: 0.3975 - acc: 0.9348
Epoch 43/300
230/230 [==============================] - 0s 457us/step - loss: 0.4145 - acc: 0.8913
Epoch 44/300
230/230 [==============================] - 0s 400us/step - loss: 0.4240 - acc: 0.8957
Epoch 45/300
230/230 [==============================] - 0s 452us/step - loss: 0.4118 - acc: 0.9087
Epoch 46/300
230/230 [==============================] - 0s 465us/step - loss: 0.3616 - acc: 0.9435
Epoch 47/300
230/230 [==============================] - 0s 439us/step - loss: 0.3851 - acc: 0.9261
Epoch 48/300
230/230 [==============================] - 0s 422us/step - loss: 0.4007 - acc: 0.9043
Epoch 49/300
230/230 [==============================] - 0s 422us/step - loss: 0.3639 - acc: 0.9304
Epoch 50/300
230/230 [==============================] - 0s 430us/step - loss: 0.3825 - acc: 0.9130
Epoch 51/300
230/230 [==============================] - ETA: 0s - loss: 0.3280 - acc: 0.942 - 0s 396us/step - loss: 0.3445 - acc: 0.9391
Epoch 52/300
230/230 [==============================] - 0s 496us/step - loss: 0.3671 - acc: 0.9174
Epoch 53/300
230/230 [==============================] - 0s 422us/step - loss: 0.3913 - acc: 0.9130
Epoch 54/300
230/230 [==============================] - 0s 409us/step - loss: 0.3456 - acc: 0.9391
Epoch 55/300
230/230 [==============================] - 0s 439us/step - loss: 0.3473 - acc: 0.9261
Epoch 56/300
230/230 [==============================] - 0s 430us/step - loss: 0.3798 - acc: 0.9087
Epoch 57/300
230/230 [==============================] - 0s 452us/step - loss: 0.3603 - acc: 0.9261
Epoch 58/300
230/230 [==============================] - 0s 457us/step - loss: 0.3569 - acc: 0.9174
Epoch 59/300
230/230 [==============================] - 0s 387us/step - loss: 0.3761 - acc: 0.9391
Epoch 60/300
230/230 [==============================] - 0s 431us/step - loss: 0.3432 - acc: 0.9261
Epoch 61/300
230/230 [==============================] - 0s 457us/step - loss: 0.3364 - acc: 0.9435
Epoch 62/300
230/230 [==============================] - 0s 431us/step - loss: 0.3409 - acc: 0.9261
Epoch 63/300
230/230 [==============================] - 0s 426us/step - loss: 0.3301 - acc: 0.9478
Epoch 64/300
230/230 [==============================] - 0s 426us/step - loss: 0.3628 - acc: 0.9217
Epoch 65/300
230/230 [==============================] - 0s 413us/step - loss: 0.3292 - acc: 0.9478
Epoch 66/300
230/230 [==============================] - 0s 435us/step - loss: 0.3390 - acc: 0.9348
Epoch 67/300
230/230 [==============================] - 0s 478us/step - loss: 0.3176 - acc: 0.9435 0s - loss: 0.3313 - acc: 0.937
Epoch 68/300
230/230 [==============================] - 0s 404us/step - loss: 0.3428 - acc: 0.9348
Epoch 69/300
230/230 [==============================] - 0s 448us/step - loss: 0.3391 - acc: 0.9348
Epoch 70/300
230/230 [==============================] - 0s 387us/step - loss: 0.3360 - acc: 0.9261
Epoch 71/300
230/230 [==============================] - 0s 465us/step - loss: 0.3253 - acc: 0.9478
Epoch 72/300
230/230 [==============================] - 0s 404us/step - loss: 0.3269 - acc: 0.9304
Epoch 73/300
230/230 [==============================] - 0s 465us/step - loss: 0.3082 - acc: 0.9522
Epoch 74/300
230/230 [==============================] - 0s 430us/step - loss: 0.3118 - acc: 0.9391
Epoch 75/300
230/230 [==============================] - 0s 404us/step - loss: 0.3272 - acc: 0.9435
Epoch 76/300
230/230 [==============================] - 0s 443us/step - loss: 0.2976 - acc: 0.9522
Epoch 77/300
230/230 [==============================] - 0s 439us/step - loss: 0.3373 - acc: 0.9217
Epoch 78/300
230/230 [==============================] - 0s 396us/step - loss: 0.3274 - acc: 0.9304
Epoch 79/300
230/230 [==============================] - 0s 448us/step - loss: 0.3525 - acc: 0.9348
Epoch 80/300
230/230 [==============================] - 0s 396us/step - loss: 0.3185 - acc: 0.9348
Epoch 81/300
230/230 [==============================] - 0s 439us/step - loss: 0.3026 - acc: 0.9609
Epoch 82/300
230/230 [==============================] - 0s 396us/step - loss: 0.3128 - acc: 0.9391
Epoch 83/300
230/230 [==============================] - 0s 417us/step - loss: 0.3082 - acc: 0.9391
Epoch 84/300
230/230 [==============================] - 0s 417us/step - loss: 0.3270 - acc: 0.9261
Epoch 85/300
230/230 [==============================] - 0s 409us/step - loss: 0.3031 - acc: 0.9522
Epoch 86/300
230/230 [==============================] - 0s 461us/step - loss: 0.3144 - acc: 0.9435
Epoch 87/300
230/230 [==============================] - 0s 439us/step - loss: 0.3036 - acc: 0.9391
Epoch 88/300
230/230 [==============================] - 0s 478us/step - loss: 0.2779 - acc: 0.9609
Epoch 89/300
230/230 [==============================] - 0s 439us/step - loss: 0.3023 - acc: 0.9435
Epoch 90/300
230/230 [==============================] - 0s 630us/step - loss: 0.2833 - acc: 0.9609
Epoch 91/300
230/230 [==============================] - 0s 474us/step - loss: 0.3100 - acc: 0.9391
Epoch 92/300
230/230 [==============================] - 0s 452us/step - loss: 0.3067 - acc: 0.9348
Epoch 93/300
230/230 [==============================] - 0s 513us/step - loss: 0.2941 - acc: 0.9435
Epoch 94/300
230/230 [==============================] - 0s 448us/step - loss: 0.3255 - acc: 0.9261
Epoch 95/300
230/230 [==============================] - 0s 496us/step - loss: 0.2581 - acc: 0.9522
Epoch 96/300
230/230 [==============================] - 0s 430us/step - loss: 0.2872 - acc: 0.9478
Epoch 97/300
230/230 [==============================] - 0s 431us/step - loss: 0.2837 - acc: 0.9435
Epoch 98/300
230/230 [==============================] - 0s 431us/step - loss: 0.2874 - acc: 0.9522
Epoch 99/300
230/230 [==============================] - 0s 870us/step - loss: 0.3129 - acc: 0.9217 0s - loss: 0.3172 - acc: 0.919
Epoch 100/300
230/230 [==============================] - 0s 435us/step - loss: 0.2731 - acc: 0.9391
Epoch 101/300
230/230 [==============================] - 0s 465us/step - loss: 0.2640 - acc: 0.9609
Epoch 102/300
230/230 [==============================] - 0s 413us/step - loss: 0.2887 - acc: 0.9435
Epoch 103/300
230/230 [==============================] - 0s 461us/step - loss: 0.2841 - acc: 0.9478
Epoch 104/300
230/230 [==============================] - 0s 443us/step - loss: 0.2797 - acc: 0.9565
Epoch 105/300
230/230 [==============================] - 0s 474us/step - loss: 0.2681 - acc: 0.9565
Epoch 106/300
230/230 [==============================] - 0s 430us/step - loss: 0.2673 - acc: 0.9609
Epoch 107/300
230/230 [==============================] - 0s 496us/step - loss: 0.2826 - acc: 0.9565
Epoch 108/300
230/230 [==============================] - 0s 470us/step - loss: 0.3363 - acc: 0.9261
Epoch 109/300
230/230 [==============================] - 0s 426us/step - loss: 0.2657 - acc: 0.9565
Epoch 110/300
230/230 [==============================] - 0s 578us/step - loss: 0.2469 - acc: 0.9652
Epoch 111/300
230/230 [==============================] - 0s 430us/step - loss: 0.2600 - acc: 0.9609
Epoch 112/300
230/230 [==============================] - 0s 422us/step - loss: 0.2616 - acc: 0.9478
Epoch 113/300
230/230 [==============================] - 0s 478us/step - loss: 0.2660 - acc: 0.9435
Epoch 114/300
230/230 [==============================] - 0s 443us/step - loss: 0.2977 - acc: 0.9391
Epoch 115/300
230/230 [==============================] - 0s 422us/step - loss: 0.2580 - acc: 0.9522
Epoch 116/300
230/230 [==============================] - 0s 426us/step - loss: 0.2826 - acc: 0.9478
Epoch 117/300
230/230 [==============================] - 0s 465us/step - loss: 0.2629 - acc: 0.9478
Epoch 118/300
230/230 [==============================] - 0s 435us/step - loss: 0.2665 - acc: 0.9522
Epoch 119/300
230/230 [==============================] - 0s 443us/step - loss: 0.2682 - acc: 0.9348
Epoch 120/300
230/230 [==============================] - 0s 435us/step - loss: 0.2614 - acc: 0.9522
Epoch 121/300
230/230 [==============================] - 0s 443us/step - loss: 0.2354 - acc: 0.9696
Epoch 122/300
230/230 [==============================] - 0s 417us/step - loss: 0.2588 - acc: 0.9478
Epoch 123/300
230/230 [==============================] - 0s 422us/step - loss: 0.2602 - acc: 0.9435
Epoch 124/300
230/230 [==============================] - 0s 430us/step - loss: 0.2753 - acc: 0.9435
Epoch 125/300
230/230 [==============================] - 0s 443us/step - loss: 0.2695 - acc: 0.9478
Epoch 126/300
230/230 [==============================] - 0s 443us/step - loss: 0.2482 - acc: 0.9696
Epoch 127/300
230/230 [==============================] - 0s 474us/step - loss: 0.2547 - acc: 0.9478
Epoch 128/300
230/230 [==============================] - 0s 448us/step - loss: 0.2590 - acc: 0.9478
Epoch 129/300
230/230 [==============================] - 0s 435us/step - loss: 0.2298 - acc: 0.9696
Epoch 130/300
230/230 [==============================] - 0s 426us/step - loss: 0.2462 - acc: 0.9565
Epoch 131/300
230/230 [==============================] - 0s 435us/step - loss: 0.2507 - acc: 0.9696
Epoch 132/300
230/230 [==============================] - 0s 417us/step - loss: 0.2758 - acc: 0.9478
Epoch 133/300
230/230 [==============================] - 0s 417us/step - loss: 0.2407 - acc: 0.9609
Epoch 134/300
230/230 [==============================] - 0s 426us/step - loss: 0.2675 - acc: 0.9522
Epoch 135/300
230/230 [==============================] - 0s 417us/step - loss: 0.2547 - acc: 0.9652
Epoch 136/300
230/230 [==============================] - 0s 457us/step - loss: 0.2666 - acc: 0.9522
Epoch 137/300
230/230 [==============================] - 0s 461us/step - loss: 0.2427 - acc: 0.9652
Epoch 138/300
230/230 [==============================] - 0s 439us/step - loss: 0.2551 - acc: 0.9478
Epoch 139/300
230/230 [==============================] - 0s 457us/step - loss: 0.2425 - acc: 0.9652
Epoch 140/300
230/230 [==============================] - 0s 400us/step - loss: 0.2337 - acc: 0.9565
Epoch 141/300
230/230 [==============================] - 0s 435us/step - loss: 0.2534 - acc: 0.9652
Epoch 142/300
230/230 [==============================] - 0s 417us/step - loss: 0.2534 - acc: 0.9522
Epoch 143/300
230/230 [==============================] - 0s 435us/step - loss: 0.2706 - acc: 0.9391
Epoch 144/300
230/230 [==============================] - 0s 430us/step - loss: 0.2323 - acc: 0.9565
Epoch 145/300
230/230 [==============================] - 0s 461us/step - loss: 0.2357 - acc: 0.9565
Epoch 146/300
230/230 [==============================] - 0s 496us/step - loss: 0.2426 - acc: 0.9478
Epoch 147/300
230/230 [==============================] - 0s 448us/step - loss: 0.2317 - acc: 0.9696
Epoch 148/300
230/230 [==============================] - 0s 409us/step - loss: 0.2345 - acc: 0.9652
Epoch 149/300
230/230 [==============================] - 0s 474us/step - loss: 0.2267 - acc: 0.9783
Epoch 150/300
230/230 [==============================] - 0s 496us/step - loss: 0.2694 - acc: 0.9565 0s - loss: 0.2881 - acc: 0.950
Epoch 151/300
230/230 [==============================] - 0s 500us/step - loss: 0.2189 - acc: 0.9696
Epoch 152/300
230/230 [==============================] - 0s 522us/step - loss: 0.2224 - acc: 0.9696
Epoch 153/300
230/230 [==============================] - 0s 678us/step - loss: 0.2487 - acc: 0.9522
Epoch 154/300
230/230 [==============================] - 0s 465us/step - loss: 0.2262 - acc: 0.9652
Epoch 155/300
230/230 [==============================] - 0s 465us/step - loss: 0.2135 - acc: 0.9696
Epoch 156/300
230/230 [==============================] - 0s 417us/step - loss: 0.2279 - acc: 0.9739
Epoch 157/300
230/230 [==============================] - 0s 543us/step - loss: 0.2295 - acc: 0.9522
Epoch 158/300
230/230 [==============================] - 0s 426us/step - loss: 0.2213 - acc: 0.9652
Epoch 159/300
230/230 [==============================] - 0s 457us/step - loss: 0.2327 - acc: 0.9478
Epoch 160/300
230/230 [==============================] - 0s 444us/step - loss: 0.2372 - acc: 0.9696
Epoch 161/300
230/230 [==============================] - 0s 439us/step - loss: 0.2362 - acc: 0.9565
Epoch 162/300
230/230 [==============================] - 0s 452us/step - loss: 0.2348 - acc: 0.9609
Epoch 163/300
230/230 [==============================] - 0s 435us/step - loss: 0.2358 - acc: 0.9565
Epoch 164/300
230/230 [==============================] - 0s 457us/step - loss: 0.2316 - acc: 0.9522
Epoch 165/300
230/230 [==============================] - 0s 430us/step - loss: 0.2261 - acc: 0.9696
Epoch 166/300
230/230 [==============================] - 0s 504us/step - loss: 0.2155 - acc: 0.9739
Epoch 167/300
230/230 [==============================] - 0s 487us/step - loss: 0.2415 - acc: 0.9565
Epoch 168/300
230/230 [==============================] - 0s 504us/step - loss: 0.2075 - acc: 0.9739
Epoch 169/300
230/230 [==============================] - 0s 474us/step - loss: 0.2118 - acc: 0.9652
Epoch 170/300
230/230 [==============================] - 0s 461us/step - loss: 0.2355 - acc: 0.9652
Epoch 171/300
230/230 [==============================] - 0s 461us/step - loss: 0.2524 - acc: 0.9304
Epoch 172/300
230/230 [==============================] - 0s 504us/step - loss: 0.2186 - acc: 0.9652
Epoch 173/300
230/230 [==============================] - 0s 652us/step - loss: 0.2130 - acc: 0.9696
Epoch 174/300
230/230 [==============================] - 0s 552us/step - loss: 0.2151 - acc: 0.9565
Epoch 175/300
230/230 [==============================] - 0s 743us/step - loss: 0.2393 - acc: 0.9565
Epoch 176/300
230/230 [==============================] - 0s 496us/step - loss: 0.2270 - acc: 0.9522
Epoch 177/300
230/230 [==============================] - 0s 448us/step - loss: 0.2263 - acc: 0.9696
Epoch 178/300
230/230 [==============================] - 0s 487us/step - loss: 0.2172 - acc: 0.9609
Epoch 179/300
230/230 [==============================] - 0s 530us/step - loss: 0.2226 - acc: 0.9565
Epoch 180/300
230/230 [==============================] - 0s 513us/step - loss: 0.2246 - acc: 0.9565
Epoch 181/300
230/230 [==============================] - 0s 448us/step - loss: 0.2422 - acc: 0.9565
Epoch 182/300
230/230 [==============================] - 0s 774us/step - loss: 0.2060 - acc: 0.9826
Epoch 183/300
230/230 [==============================] - 0s 857us/step - loss: 0.2153 - acc: 0.9522
Epoch 184/300
230/230 [==============================] - 0s 561us/step - loss: 0.2244 - acc: 0.9478
Epoch 185/300
230/230 [==============================] - 0s 983us/step - loss: 0.2156 - acc: 0.9783
Epoch 186/300
230/230 [==============================] - 0s 922us/step - loss: 0.2097 - acc: 0.9739
Epoch 187/300
230/230 [==============================] - 0s 670us/step - loss: 0.2242 - acc: 0.9696
Epoch 188/300
230/230 [==============================] - 0s 465us/step - loss: 0.2360 - acc: 0.9522
Epoch 189/300
230/230 [==============================] - 0s 439us/step - loss: 0.2312 - acc: 0.9565
Epoch 190/300
230/230 [==============================] - 0s 478us/step - loss: 0.1991 - acc: 0.9739
Epoch 191/300
230/230 [==============================] - 0s 543us/step - loss: 0.2273 - acc: 0.9522
Epoch 192/300
230/230 [==============================] - 0s 474us/step - loss: 0.2038 - acc: 0.9783
Epoch 193/300
230/230 [==============================] - 0s 478us/step - loss: 0.2092 - acc: 0.9696
Epoch 194/300
230/230 [==============================] - 0s 422us/step - loss: 0.2108 - acc: 0.9609
Epoch 195/300
230/230 [==============================] - 0s 430us/step - loss: 0.2181 - acc: 0.9609
Epoch 196/300
230/230 [==============================] - 0s 452us/step - loss: 0.2077 - acc: 0.9652
Epoch 197/300
230/230 [==============================] - 0s 574us/step - loss: 0.1911 - acc: 0.9826
Epoch 198/300
230/230 [==============================] - 0s 648us/step - loss: 0.2032 - acc: 0.9652
Epoch 199/300
230/230 [==============================] - 0s 517us/step - loss: 0.1967 - acc: 0.9739
Epoch 200/300
230/230 [==============================] - 0s 535us/step - loss: 0.2231 - acc: 0.9522
Epoch 201/300
230/230 [==============================] - 0s 670us/step - loss: 0.2330 - acc: 0.9435
Epoch 202/300
230/230 [==============================] - 0s 626us/step - loss: 0.1996 - acc: 0.9739
Epoch 203/300
230/230 [==============================] - 0s 561us/step - loss: 0.1934 - acc: 0.9739
Epoch 204/300
230/230 [==============================] - 0s 487us/step - loss: 0.2007 - acc: 0.9609
Epoch 205/300
230/230 [==============================] - 0s 561us/step - loss: 0.2012 - acc: 0.9696
Epoch 206/300
230/230 [==============================] - ETA: 0s - loss: 0.2140 - acc: 0.962 - 0s 530us/step - loss: 0.2173 - acc: 0.9609
Epoch 207/300
230/230 [==============================] - 0s 570us/step - loss: 0.2177 - acc: 0.9609
Epoch 208/300
230/230 [==============================] - 0s 743us/step - loss: 0.2111 - acc: 0.9652
Epoch 209/300
230/230 [==============================] - 0s 526us/step - loss: 0.1856 - acc: 0.9826
Epoch 210/300
230/230 [==============================] - 0s 726us/step - loss: 0.1875 - acc: 0.9826
Epoch 211/300
230/230 [==============================] - 0s 535us/step - loss: 0.1925 - acc: 0.9783
Epoch 212/300
230/230 [==============================] - 0s 500us/step - loss: 0.1968 - acc: 0.9783
Epoch 213/300
230/230 [==============================] - 0s 548us/step - loss: 0.2066 - acc: 0.9652
Epoch 214/300
230/230 [==============================] - 0s 552us/step - loss: 0.2120 - acc: 0.9652
Epoch 215/300
230/230 [==============================] - 0s 670us/step - loss: 0.2154 - acc: 0.9522
Epoch 216/300
230/230 [==============================] - 0s 657us/step - loss: 0.2156 - acc: 0.9696
Epoch 217/300
230/230 [==============================] - ETA: 0s - loss: 0.2285 - acc: 0.956 - 0s 574us/step - loss: 0.2249 - acc: 0.9565
Epoch 218/300
230/230 [==============================] - 0s 574us/step - loss: 0.2064 - acc: 0.9696
Epoch 219/300
230/230 [==============================] - 0s 487us/step - loss: 0.2077 - acc: 0.9565
Epoch 220/300
230/230 [==============================] - 0s 704us/step - loss: 0.1884 - acc: 0.9783
Epoch 221/300
230/230 [==============================] - 0s 596us/step - loss: 0.2150 - acc: 0.9739
Epoch 222/300
230/230 [==============================] - 0s 735us/step - loss: 0.1805 - acc: 0.9870
Epoch 223/300
230/230 [==============================] - 0s 448us/step - loss: 0.1992 - acc: 0.9783
Epoch 224/300
230/230 [==============================] - 0s 461us/step - loss: 0.2030 - acc: 0.9609
Epoch 225/300
230/230 [==============================] - 0s 457us/step - loss: 0.1900 - acc: 0.9739
Epoch 226/300
230/230 [==============================] - 0s 465us/step - loss: 0.1940 - acc: 0.9696
Epoch 227/300
230/230 [==============================] - 0s 417us/step - loss: 0.1698 - acc: 0.9870
Epoch 228/300
230/230 [==============================] - 0s 435us/step - loss: 0.1962 - acc: 0.9696
Epoch 229/300
230/230 [==============================] - 0s 448us/step - loss: 0.1773 - acc: 0.9913
Epoch 230/300
230/230 [==============================] - 0s 430us/step - loss: 0.1940 - acc: 0.9739
Epoch 231/300
230/230 [==============================] - 0s 457us/step - loss: 0.2065 - acc: 0.9652
Epoch 232/300
230/230 [==============================] - 0s 452us/step - loss: 0.1805 - acc: 0.9826
Epoch 233/300
230/230 [==============================] - 0s 417us/step - loss: 0.1873 - acc: 0.9783
Epoch 234/300
230/230 [==============================] - 0s 448us/step - loss: 0.2002 - acc: 0.9652
Epoch 235/300
230/230 [==============================] - 0s 443us/step - loss: 0.1808 - acc: 0.9696
Epoch 236/300
230/230 [==============================] - 0s 435us/step - loss: 0.1763 - acc: 0.9913
Epoch 237/300
230/230 [==============================] - 0s 400us/step - loss: 0.1736 - acc: 0.9826
Epoch 238/300
230/230 [==============================] - 0s 430us/step - loss: 0.1673 - acc: 0.9870
Epoch 239/300
230/230 [==============================] - 0s 443us/step - loss: 0.1674 - acc: 0.9870
Epoch 240/300
230/230 [==============================] - 0s 474us/step - loss: 0.1640 - acc: 0.9870
Epoch 241/300
230/230 [==============================] - 0s 530us/step - loss: 0.1966 - acc: 0.9696
Epoch 242/300
230/230 [==============================] - 0s 683us/step - loss: 0.1882 - acc: 0.9696
Epoch 243/300
230/230 [==============================] - 0s 578us/step - loss: 0.1693 - acc: 0.9913
Epoch 244/300
230/230 [==============================] - 0s 583us/step - loss: 0.1768 - acc: 0.9826
Epoch 245/300
230/230 [==============================] - 0s 487us/step - loss: 0.1939 - acc: 0.9739
Epoch 246/300
230/230 [==============================] - 0s 504us/step - loss: 0.1832 - acc: 0.9696
Epoch 247/300
230/230 [==============================] - 0s 452us/step - loss: 0.1688 - acc: 0.9870
Epoch 248/300
230/230 [==============================] - 0s 435us/step - loss: 0.1801 - acc: 0.9739
Epoch 249/300
230/230 [==============================] - 0s 500us/step - loss: 0.1718 - acc: 0.9739
Epoch 250/300
230/230 [==============================] - 0s 504us/step - loss: 0.1707 - acc: 0.9870
Epoch 251/300
230/230 [==============================] - 0s 574us/step - loss: 0.1799 - acc: 0.9739
Epoch 252/300
230/230 [==============================] - 0s 517us/step - loss: 0.1695 - acc: 0.9870
Epoch 253/300
230/230 [==============================] - 0s 491us/step - loss: 0.1763 - acc: 0.9652
Epoch 254/300
230/230 [==============================] - 0s 491us/step - loss: 0.1745 - acc: 0.9696
Epoch 255/300
230/230 [==============================] - 0s 470us/step - loss: 0.1833 - acc: 0.9696
Epoch 256/300
230/230 [==============================] - 0s 483us/step - loss: 0.1911 - acc: 0.9696
Epoch 257/300
230/230 [==============================] - 0s 500us/step - loss: 0.1682 - acc: 0.9826
Epoch 258/300
230/230 [==============================] - 0s 522us/step - loss: 0.1695 - acc: 0.9826
Epoch 259/300
230/230 [==============================] - 0s 513us/step - loss: 0.1801 - acc: 0.9739
Epoch 260/300
230/230 [==============================] - 0s 613us/step - loss: 0.1885 - acc: 0.9696
Epoch 261/300
230/230 [==============================] - 0s 487us/step - loss: 0.1797 - acc: 0.9826
Epoch 262/300
230/230 [==============================] - 0s 726us/step - loss: 0.1699 - acc: 0.9783
Epoch 263/300
230/230 [==============================] - 0s 687us/step - loss: 0.1725 - acc: 0.9826
Epoch 264/300
230/230 [==============================] - 0s 543us/step - loss: 0.1663 - acc: 0.9870
Epoch 265/300
230/230 [==============================] - 0s 570us/step - loss: 0.1754 - acc: 0.9696
Epoch 266/300
230/230 [==============================] - 0s 583us/step - loss: 0.1704 - acc: 0.9826
Epoch 267/300
230/230 [==============================] - 0s 809us/step - loss: 0.1781 - acc: 0.9739
Epoch 268/300
230/230 [==============================] - 0s 557us/step - loss: 0.1695 - acc: 0.9913
Epoch 269/300
230/230 [==============================] - 0s 474us/step - loss: 0.1874 - acc: 0.9826
Epoch 270/300
230/230 [==============================] - 0s 530us/step - loss: 0.1921 - acc: 0.9783
Epoch 271/300
230/230 [==============================] - 0s 517us/step - loss: 0.1701 - acc: 0.9826
Epoch 272/300
230/230 [==============================] - 0s 470us/step - loss: 0.1752 - acc: 0.9696
Epoch 273/300
230/230 [==============================] - 0s 448us/step - loss: 0.1783 - acc: 0.9783
Epoch 274/300
230/230 [==============================] - 0s 483us/step - loss: 0.1765 - acc: 0.9826
Epoch 275/300
230/230 [==============================] - 0s 600us/step - loss: 0.1663 - acc: 0.9826
Epoch 276/300
230/230 [==============================] - 0s 526us/step - loss: 0.1818 - acc: 0.9783
Epoch 277/300
230/230 [==============================] - 0s 448us/step - loss: 0.1815 - acc: 0.9783
Epoch 278/300
230/230 [==============================] - 0s 517us/step - loss: 0.1774 - acc: 0.9826
Epoch 279/300
230/230 [==============================] - 0s 809us/step - loss: 0.1802 - acc: 0.9696
Epoch 280/300
230/230 [==============================] - 0s 457us/step - loss: 0.1696 - acc: 0.9870
Epoch 281/300
230/230 [==============================] - 0s 470us/step - loss: 0.1688 - acc: 0.9826
Epoch 282/300
230/230 [==============================] - 0s 513us/step - loss: 0.1797 - acc: 0.9783 0s - loss: 0.1750 - acc: 0.987
Epoch 283/300
230/230 [==============================] - 0s 470us/step - loss: 0.1441 - acc: 0.9957
Epoch 284/300
230/230 [==============================] - 0s 543us/step - loss: 0.1840 - acc: 0.9609
Epoch 285/300
230/230 [==============================] - 0s 491us/step - loss: 0.1788 - acc: 0.9739
Epoch 286/300
230/230 [==============================] - 0s 630us/step - loss: 0.1634 - acc: 0.9783
Epoch 287/300
230/230 [==============================] - 0s 478us/step - loss: 0.1632 - acc: 0.9783
Epoch 288/300
230/230 [==============================] - 0s 465us/step - loss: 0.1607 - acc: 0.9870
Epoch 289/300
230/230 [==============================] - 0s 1ms/step - loss: 0.1846 - acc: 0.9652
Epoch 290/300
230/230 [==============================] - 0s 609us/step - loss: 0.1685 - acc: 0.9826
Epoch 291/300
230/230 [==============================] - 0s 526us/step - loss: 0.1799 - acc: 0.9783
Epoch 292/300
230/230 [==============================] - 0s 487us/step - loss: 0.1494 - acc: 0.9826
Epoch 293/300
230/230 [==============================] - 0s 517us/step - loss: 0.1812 - acc: 0.9739
Epoch 294/300
230/230 [==============================] - 0s 570us/step - loss: 0.1763 - acc: 0.9696
Epoch 295/300
230/230 [==============================] - 0s 509us/step - loss: 0.1724 - acc: 0.9652
Epoch 296/300
230/230 [==============================] - 0s 535us/step - loss: 0.1673 - acc: 0.9739
Epoch 297/300
230/230 [==============================] - 0s 478us/step - loss: 0.1819 - acc: 0.9696
Epoch 298/300
230/230 [==============================] - 0s 457us/step - loss: 0.1642 - acc: 0.9870
Epoch 299/300
230/230 [==============================] - ETA: 0s - loss: 0.1451 - acc: 0.984 - 0s 661us/step - loss: 0.1600 - acc: 0.9696
Epoch 300/300
230/230 [==============================] - 0s 643us/step - loss: 0.1615 - acc: 0.9870
"""

epochs, loss_values, accuracy_values = analyze_log(log)

# Plot the loss values
plt.figure(figsize=(8, 4))
plt.plot(epochs, loss_values)
plt.title('Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.show()

# Plot the accuracy values
plt.figure(figsize=(8, 4))
plt.plot(epochs, accuracy_values)
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()