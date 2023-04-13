import tensorflow as tf
import keras.backend as K
import numpy as np
import pandas as pd
from tensorflow.python import keras
import pickle as pkl
import matplotlib.pyplot as plt

from keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from keras.models import Sequential
from keras.layers import Lambda
from keras.layers import MaxPooling1D
from keras.layers import Dense
from keras.layers import Conv1D
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy


with open('CNN_input_test1.pickle', 'rb') as f:
    X = pkl.load(f)

tf.convert_to_tensor (X)

DF_Y = pd.read_csv("CNN_output_test1.csv")

DF_Y ['Rifampicin'] = DF_Y ['Rifampicin'].replace({'S': 0, 'R': 1}, regex=True)
DF_Y ['Isoniazid'] = DF_Y ['Isoniazid'].replace({'S': 0, 'R': 1}, regex=True)
DF_Y['Ethambutol'] = DF_Y ['Ethambutol'].replace({'S': 0, 'R': 1}, regex=True)
DF_Y ['Pyrazinamide'] = DF_Y ['Pyrazinamide'].replace({'S': 0, 'R': 1}, regex=True)

Y = DF_Y.values
print (Y)

def output_division (list, Ethambutol, Isoniazid, Pyrazinamide, Rifampicin):

    for i in list:
        Ethambutol.append(i[0])
        Isoniazid.append(i[1])
        Pyrazinamide.append(i[2])
        Rifampicin.append(i[3])
    
    print (Ethambutol)
    print (len(Isoniazid))
    print (len(Pyrazinamide))
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
def get_threshold(y_valid, y_pred_ROC):
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

skf = KFold(n_splits=10, shuffle=True, random_state=42)

fold_scores = []
confusion_matrices = []

for fold, (train_index, test_index) in enumerate(skf.split(X, Y)):
    print(f'Fold {fold+1}:')
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = Y[train_index], Y[test_index]

    # Build the model
    model = create_model()

    # Train the model
    model.fit(X_train, Y_train, epochs=10, batch_size=32)

    # Evaluate the model on the test data
    y_pred = model.predict(X_test)
    y_pred_binary = (y_pred > 0.5).astype(int) # convert probabilities to binary predictions
    confusion_matrices.append([confusion_matrix(Y_test[i:], y_pred_binary[i:]) for i in range(4)])

    # Convert Y_test back to multilabel-indicator format for evaluation
    scores = model.evaluate(X_test, Y_test, verbose=0)
    fold_scores.append(scores)


print('Confusion matrices:')
for i, loop in range(4):
    print(f'Output {i+1} confusion matrix:')
    combined_cm = np.sum([cm[i] for cm in confusion_matrices], axis=0)
    print(combined_cm)

    TP, FP, TN, FN = (combined_cm[1][1], combined_cm[0][1], combined_cm[0][0], combined_cm[1][0])
    print (TP, FP, TN, FN)   

    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    Sensitivity = TP / (FN + TP)
    Specificity = TN / (FP + TN)

    print ('The CV Accuracy on loop number' + {loop} + str(Accuracy))
    print ('The CV Sensitivity on loop number ' + {loop} + str(Sensitivity))
    print ('The CV Specificity on loop number ' + {loop} + str(Specificity))

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.6, random_state=42)
X_test, X_valid, Y_test, Y_valid= train_test_split(X_test, Y_test, test_size=0.5, random_state=42)

model = create_model()

earlystopping = EarlyStopping(monitor='val_loss', patience=1)
model.fit(X_train, Y_train, validation_data=(X_valid, Y_valid), epochs=250, callbacks=earlystopping)

plt.figure()
plt.plot(model.history.history['loss'])
plt.plot(model.history.history['val_loss'])

plt.title('Model_loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'])

plt.figure()
plt.plot(model.history.history['accuracy'])
plt.plot(model.history.history['val_accuracy'])

plt.title('Model_accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'])

plt.show()

valid_output = model.predict(X_valid)
valid_binary_output = (valid_output > 0.5).astype(int)

Eth_val_model = []
Iso_val_model = []
Pyr_val_model = []
Rif_val_model = []

Eth_val_model, Iso_val_model, Pyr_val_model, Rif_val_model = output_division (valid_binary_output, Eth_val_model, Iso_val_model, Pyr_val_model, Rif_val_model)

Eth_valid = []
Iso_valid = []
Pyr_valid = []
Rif_valid = []

Eth_valid, Iso_valid, Pyr_valid, Rif_valid = output_division (Y_valid, Eth_valid, Iso_valid, Pyr_valid, Rif_valid)

confusion_matrix_valid (Eth_valid, Eth_val_model)
confusion_matrix_valid (Iso_valid, Iso_val_model)
confusion_matrix_valid (Pyr_valid, Pyr_val_model)
confusion_matrix_valid (Rif_valid, Rif_val_model)

Eth_threshold, Eth_fpr, Eth_tpr = get_threshold(Eth_valid, Eth_val_model)
Iso_threshold, Iso_fpr, Iso_tpr = get_threshold(Iso_valid, Iso_val_model)
Pyr_threshold, Pyr_fpr, Pyr_tpr = get_threshold(Pyr_valid, Pyr_val_model)
Rif_threshold, Rif_fpr, Rif_tpr = get_threshold(Rif_valid, Rif_val_model)


test_output = model.predict(X_test)

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


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8, random_state=42)
X_test, X_valid, Y_test, Y_valid= train_test_split(X_test, Y_test, test_size=0.5, random_state=42)

earlystopping = EarlyStopping(monitor='val_loss', patience=10)
model.fit(X_train, Y_train, validation_data=(X_valid, Y_valid), epochs=250, batch_size=128, callbacks=earlystopping)

