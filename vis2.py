import os
import re
import matplotlib.pyplot as plt
log = os.getcwd() + '\outputs\\transformerWithpretrain\Markdown\\transformerWithpretrain_1.md'

#log = os.getcwd() + '\outputs\Transformer\Markdown\Transformer.md'
#log  = os.getcwd() + '/outputs/CNN/Markdown/CNN_0.md' 
#log  = os.getcwd() + '/outputs/LSTMTest2withSoftmax/Markdown/LSTMTest2withSoftmax_0.md'
#read log
with open(log, 'r') as f:
    log = f.read()
# Extract the test accuracy, train loss, test loss, and train accuracy values using regular expression
test_accuracy_values = re.findall(r"Test accuracy is: (\d+\.\d+)", log)[:100]
train_loss_values = re.findall(r"Train loss: (\d+\.\d+)", log)[:100]
test_loss_values = re.findall(r"Test loss: (\d+\.\d+)", log)[:100]
train_accuracy_values = re.findall(r"Train accuracy: (\d+\.\d+)", log)[:100]
print("test_accuracy_values", test_accuracy_values)
print("train_loss_values", train_loss_values)
print("test_loss_values", test_loss_values)
print("train_accuracy_values", train_accuracy_values)
# Convert the test accuracy, train loss, test loss, and train accuracy values to float
test_accuracy_values = [float(value) for value in test_accuracy_values]
train_loss_values = [float(value) for value in train_loss_values]
test_loss_values = [float(value) for value in test_loss_values]
train_accuracy_values = [float(value) for value in train_accuracy_values]

# Calculate the average test accuracy
#average_test_accuracy = sum(test_accuracy_values) / len(test_accuracy_values)

# Find the maximum and minimum test accuracy values
# max_test_accuracy = max(test_accuracy_values)
# min_test_accuracy = min(test_accuracy_values)

# Find the epoch number with the highest test accuracy
#best_epoch = test_accuracy_values.index(max_test_accuracy) + 1

# Plot the train loss and test loss
epochs = range(1, len(test_accuracy_values) + 1)

plt.figure(figsize=(10, 5))
plt.plot(epochs, train_loss_values, label='Train Loss')
plt.plot(epochs, test_loss_values, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Train Loss vs. Test Loss')
plt.legend()
plt.show()

# Plot the train accuracy and test accuracy
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_accuracy_values, label='Train Accuracy')
plt.plot(epochs, test_accuracy_values, label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Train Accuracy vs. Test Accuracy')
plt.legend()
plt.show()
