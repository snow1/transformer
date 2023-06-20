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

show_result111()