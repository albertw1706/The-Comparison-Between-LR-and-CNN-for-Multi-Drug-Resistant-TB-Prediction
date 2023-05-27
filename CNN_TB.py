import tensorflow as tf
import keras.backend as K
import numpy as np
import pandas as pd
from tensorflow.python import keras
import pickle as pkl
import matplotlib.pyplot as plt
import time
import psutil

from keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Lambda
from keras.layers import MaxPooling1D
from keras.layers import Dense
from keras.layers import Conv1D
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.optimizers import Adam


with open('CNN_input.pickle', 'rb') as f:
    X = pkl.load(f)

tf.convert_to_tensor (X)

DF_Y = pd.read_csv("CNN_output.csv")

DF_Y ['Rifampicin'] = DF_Y ['Rifampicin'].replace({'S': 0, 'R': 1}, regex=True)
DF_Y ['Isoniazid'] = DF_Y ['Isoniazid'].replace({'S': 0, 'R': 1}, regex=True)
DF_Y['Ethambutol'] = DF_Y ['Ethambutol'].replace({'S': 0, 'R': 1}, regex=True)
DF_Y ['Pyrazinamide'] = DF_Y ['Pyrazinamide'].replace({'S': 0, 'R': 1}, regex=True)

Y = DF_Y.values
print (Y)

@tf.function
def weighted_bce(alpha, y_pred):
    y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())
    y_true_ = K.cast(K.greater(alpha, 0.), K.floatx())
    mask = K.cast(K.not_equal(alpha, 0.), K.floatx())
    num_not_missing = K.sum(mask, axis=-1)
    alpha = K.abs(alpha)
    bce = - alpha * y_true_ * K.log(y_pred) - (1.0 - alpha) * (1.0 - y_true_) * K.log(1.0 - y_pred)
    masked_bce = bce * mask
    return K.sum(masked_bce, axis=-1) / num_not_missing

@tf.function
def weighted_accuracy(alpha, y_pred):
    total = K.sum(K.cast(K.not_equal(alpha, 0.), K.floatx()))
    y_true_ = K.cast(K.greater(alpha, 0.), K.floatx())
    mask = K.cast(K.not_equal(alpha, 0.), K.floatx())
    correct = K.sum(K.cast(K.equal(y_true_, K.round(y_pred)), K.floatx()) * mask)
    return correct / total

def create_alpha_matrix(Y_train, Y_valid, weight=1.):
    Arr = np.concatenate((Y_train, Y_valid), axis=0)

    alphas = np.zeros(4, dtype=np.float)
    alpha_matrix = np.zeros_like(Arr, dtype=np.float)

    for drug in range(4):

        sensitive = len(np.squeeze(np.where(Arr[:, drug] == 0.)))
        resistant = len(np.squeeze(np.where(Arr[:, drug] == 1.)))
        alphas[drug] = resistant / float(resistant + sensitive)
        alpha_matrix[:, drug][np.where(Arr[:, drug] == 1.0)] = weight * alphas[drug]
        alpha_matrix[:, drug][np.where(Arr[:, drug] == 0.0)] = - alphas[drug]

    arr_valid_row = Y_train.shape[0]
    Y_a_train, Y_a_valid = np.split(alpha_matrix, [arr_valid_row])

    return Y_a_train, Y_a_valid

def output_division (list, Ethambutol, Isoniazid, Pyrazinamide, Rifampicin):
    for i in list:
        Ethambutol.append(i[0])
        Isoniazid.append(i[1])
        Pyrazinamide.append(i[2])
        Rifampicin.append(i[3])
    
    print (Ethambutol)
    print (Isoniazid)
    print (Pyrazinamide)
    print (Rifampicin)

    return Ethambutol, Isoniazid, Pyrazinamide, Rifampicin

# Applying Threshold
def decide_output(list, threshold):
    array = np.array(list)
    array_binary = (array > threshold).astype(int)
    list_binary = array_binary.tolist()
    print(list_binary)
    return list_binary

# Confusion Matrix for valid
def confusion_matrix_valid(y_valid, y_pred):
    cm = confusion_matrix(y_valid, y_pred)
    TP, FP, TN, FN = (cm[1][1], cm[0][1], cm[0][0], cm[1][0])
    print (TP, FP, TN, FN)   

    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    Sensitivity = TP / (FN + TP)
    Specificity = TN / (FP + TN)

    print ('The Accuracy on valid set is ' + str(Accuracy))
    print ('The Sensitivity on valid set is ' + str(Sensitivity))
    print ('The Specificity on valid set is ' + str(Specificity))

# Confusion Matrix for test
def confusion_matrix_test(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    TP, FP, TN, FN = (cm[1][1], cm[0][1], cm[0][0], cm[1][0])
    print (TP, FP, TN, FN)   

    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    Sensitivity = TP / (FN + TP)
    Specificity = TN / (FP + TN)

    print ('The Accuracy on test set is ' + str(Accuracy))
    print ('The Sensitivity on test set is ' + str(Sensitivity))
    print ('The Specificity on test set is ' + str(Specificity))

# ROC and Youden for Threshold
def get_threshold_and_ROC(y_valid, y_pred_ROC):
    fpr, tpr, thresholds = roc_curve(y_valid, y_pred_ROC)

    #Youden 
    best_j = 0
    best_threshold = 0

    # Iterate over thresholds
    for i, threshold in enumerate(thresholds):
        j = tpr[i] - fpr[i]
        if j > best_j:
            best_j = j
            best_threshold = threshold
    print ((f'Best Threshold: {best_threshold:.3f}'))
    
    # Plot ROC curve
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.show()
    return best_threshold, fpr, tpr

# CNN model achitecture
def create_model ():
    #TODO: replace X.shape with passed argument
    model = Sequential()
    #TODO: add filter size argument
    model.add(Conv2D(
        64, (5, 12),
        data_format='channels_last',
        activation='relu',
        input_shape = X.shape[1:]
        ))
    model.add(Lambda(lambda x: K.squeeze(x, 1)))
    model.add(Conv1D(64, 12, activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Conv1D(32, 3, activation='relu'))
    model.add(Conv1D(32, 3, activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(4, activation='sigmoid'))

    model.compile(
        optimizer=Adam(learning_rate=np.exp(-1.0 * 9)),
        loss=weighted_bce,
        metrics=[weighted_accuracy])

    return model

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, random_state=42)
X_test, X_valid, Y_test, Y_valid= train_test_split(X_test, Y_test, test_size=0.5, random_state=42)

model = create_model()

Y_a_train, Y_a_valid = create_alpha_matrix(Y_train, Y_valid, weight=1.)

earlystopping = EarlyStopping(monitor='val_loss', patience=20)
cpu_usage_percent_list = []
start_time = time.time()
model.fit(X_train, Y_train, validation_data=(X_valid, Y_valid), epochs=250, batch_size=128, callbacks=earlystopping)
end_time = time.time()
while True:
    cpu_usage_percent = psutil.cpu_percent(interval=1)
    cpu_usage_percent_list.append(cpu_usage_percent)
    if time.time() - start_time > end_time - start_time:
        break
total_time = end_time - start_time
print(f"Total time taken: {total_time:.2f} seconds")
average_cpu_usage_percent = sum(cpu_usage_percent_list) / len(cpu_usage_percent_list)
print(f"Average CPU usage during training: {average_cpu_usage_percent:.2f}%")

plt.figure()
plt.plot(model.history.history['loss'])
plt.plot(model.history.history['val_loss'])

plt.title('Model_loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'])

plt.figure()
plt.plot(model.history.history['weighted_accuracy'])
plt.plot(model.history.history['val_weighted_accuracy'])

plt.title('Model_accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'])

plt.show()

valid_output = model.predict(X_valid)

Eth_val_model = []
Iso_val_model = []
Pyr_val_model = []
Rif_val_model = []

Eth_val_model, Iso_val_model, Pyr_val_model, Rif_val_model = output_division (valid_output, Eth_val_model, Iso_val_model, Pyr_val_model, Rif_val_model)

Eth_valid = []
Iso_valid = []
Pyr_valid = []
Rif_valid = []

Eth_valid, Iso_valid, Pyr_valid, Rif_valid = output_division (Y_valid, Eth_valid, Iso_valid, Pyr_valid, Rif_valid)

Eth_threshold, Eth_fpr, Eth_tpr = get_threshold_and_ROC(Eth_valid, Eth_val_model)
Iso_threshold, Iso_fpr, Iso_tpr = get_threshold_and_ROC(Iso_valid, Iso_val_model)
Pyr_threshold, Pyr_fpr, Pyr_tpr = get_threshold_and_ROC(Pyr_valid, Pyr_val_model)
Rif_threshold, Rif_fpr, Rif_tpr = get_threshold_and_ROC(Rif_valid, Rif_val_model)

print (Eth_threshold)
print (Iso_threshold)
print (Pyr_threshold)
print (Rif_threshold)

Eth_valid_binary = decide_output(Eth_val_model, Eth_threshold)
Iso_valid_binary = decide_output(Iso_val_model, Iso_threshold)
Pyr_valid_binary = decide_output(Pyr_val_model, Pyr_threshold)
Rif_valid_binary = decide_output(Rif_val_model, Rif_threshold)

confusion_matrix_valid (Eth_valid, Eth_valid_binary)
confusion_matrix_valid (Iso_valid, Iso_valid_binary)
confusion_matrix_valid (Pyr_valid, Pyr_valid_binary)
confusion_matrix_valid (Rif_valid, Rif_valid_binary)

skf = KFold(n_splits=10, shuffle=True, random_state=42)

fold_scores = []
confusion_matrices = []

Eth_test_list_cv = []
Eth_test_binary_list_cv = []
Iso_test_list_cv = []
Iso_test_binary_list_cv = []
Pyr_test_list_cv = []
Pyr_test_binary_list_cv = []
Rif_test_list_cv = []
Rif_test_binary_list_cv = []

for fold, (train_index, test_index) in enumerate(skf.split(X, Y)):
    print(f'Fold {fold+1}:')
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]

    # Build the model
    model = create_model()

    # Train the model
    Y_a_train, Y_a_test = create_alpha_matrix(Y_train, Y_test, weight=1.)
    earlystopping = EarlyStopping(monitor='val_loss', patience=20)
    model.fit(X_train, Y_a_train, validation_data=(X_test, Y_a_test), epochs=250, callbacks=earlystopping)

    # Evaluate the model on the test data
    y_pred = model.predict(X_test)

    Eth_test_model = []
    Iso_test_model = []
    Pyr_test_model = []
    Rif_test_model = []

    Eth_test_model, Iso_test_model, Pyr_test_model, Rif_test_model = output_division (y_pred, Eth_test_model, Iso_test_model, Pyr_test_model, Rif_test_model)

    Eth_test = []
    Iso_test = []
    Pyr_test = []
    Rif_test = []

    Eth_test, Iso_test, Pyr_test, Rif_test = output_division (Y_test, Eth_test, Iso_test, Pyr_test, Rif_test)

    Eth_test_binary = decide_output(Eth_test_model, Eth_threshold)
    Iso_test_binary = decide_output(Iso_test_model, Iso_threshold)
    Pyr_test_binary = decide_output(Pyr_test_model, Pyr_threshold)
    Rif_test_binary = decide_output(Rif_test_model, Rif_threshold)

    Eth_test_list_cv.extend(Eth_test)
    Eth_test_binary_list_cv.extend(Eth_test_binary)
    Iso_test_list_cv.extend(Iso_test)
    Iso_test_binary_list_cv.extend(Iso_test_binary)
    Pyr_test_list_cv.extend(Pyr_test)
    Pyr_test_binary_list_cv.extend(Pyr_test_binary)
    Rif_test_list_cv.extend(Rif_test)
    Rif_test_binary_list_cv.extend(Rif_test_binary)

confusion_matrix_valid(Eth_test_list_cv, Eth_test_binary_list_cv)
confusion_matrix_valid(Iso_test_list_cv, Iso_test_binary_list_cv)
confusion_matrix_valid(Pyr_test_list_cv, Pyr_test_binary_list_cv)
confusion_matrix_valid(Rif_test_list_cv, Rif_test_binary_list_cv)

cpu_usage_percent_list = []
start_time = time.time()
test_output = model.predict(X_test)
end_time = time.time()
while True:
    cpu_usage_percent = psutil.cpu_percent(interval=1)
    cpu_usage_percent_list.append(cpu_usage_percent)
    if time.time() - start_time > end_time - start_time:
        break
total_time = end_time - start_time
print(f"Total time taken: {total_time:.2f} seconds")
average_cpu_usage_percent = sum(cpu_usage_percent_list) / len(cpu_usage_percent_list)
print(f"Average CPU usage during prediction: {average_cpu_usage_percent:.2f}%")

Eth_test_model = []
Iso_test_model = []
Pyr_test_model = []
Rif_test_model = []

Eth_test_model, Iso_test_model, Pyr_test_model, Rif_test_model = output_division (test_output, Eth_test_model, Iso_test_model, Pyr_test_model, Rif_test_model)

Eth_test_binary = decide_output(Eth_test_model, Eth_threshold)
Iso_test_binary = decide_output(Iso_test_model, Iso_threshold)
Pyr_test_binary = decide_output(Pyr_test_model, Pyr_threshold)
Rif_test_binary = decide_output(Rif_test_model, Rif_threshold)

Eth_test = []
Iso_test = []
Pyr_test = []
Rif_test = []

Eth_test, Iso_test, Pyr_test, Rif_test = output_division (Y_test, Eth_test, Iso_test, Pyr_test, Rif_test)

confusion_matrix_test (Eth_test, Eth_test_binary)
confusion_matrix_test (Iso_test, Iso_test_binary)
confusion_matrix_test (Pyr_test, Pyr_test_binary)
confusion_matrix_test (Rif_test, Rif_test_binary)
