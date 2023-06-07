import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
import time
import psutil

# Open the CSV file containing input and output
DF = pd.read_csv("Model_ready.csv")

DF ['Rifampicin'] = DF ['Rifampicin'].replace({'S': 0, 'R': 1}, regex=True)
DF ['Isoniazid'] = DF ['Isoniazid'].replace({'S': 0, 'R': 1}, regex=True)
DF ['Ethambutol'] = DF ['Ethambutol'].replace({'S': 0, 'R': 1}, regex=True)
DF ['Pyrazinamide'] = DF ['Pyrazinamide'].replace({'S': 0, 'R': 1}, regex=True)

# Define the input and output
data = DF.columns.values.tolist()

exclude = ['Rifampicin', 'Isoniazid', 'Ethambutol', 'Pyrazinamide', 'Sample']
output_Rif = ['Rifampicin']
output_Iso = ['Isoniazid']
output_Eth = ['Ethambutol']
output_Pyr = ['Pyrazinamide']
input = [i for i in data if i not in exclude]

X = DF[input]
Rif = DF[output_Rif]
Iso = DF[output_Iso]
Eth = DF[output_Eth]
Pyr = DF[output_Pyr]

# Logistic regression and Confusion Matrix
def get_logistic_regression_and_CM(L2_param, solver_param, X_train, X_valid, y_train, y_valid):
    logreg = LogisticRegression(max_iter=250, C=L2_param, penalty='l2', solver=solver_param, class_weight="balanced", random_state=0)
    cpu_usage_percent_list = []
    start_time = time.time()
    logreg.fit(X_train, y_train)
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
    
    y_pred = logreg.predict(X_valid)
    y_pred_ROC = logreg.predict_proba(X_valid)[:, 1]
    print('Accuracy of logistic regression classifier on valid set: {:.2f}'.format(logreg.score(X_valid, y_valid)))
    print('Iterations needed:', np.sum(logreg.n_iter_))

    cm = confusion_matrix(y_valid, y_pred)
    TP, FP, TN, FN = (cm[1][1], cm[0][1], cm[0][0], cm[1][0])
    print (TP, FP, TN, FN)   

    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    Sensitivity = TP / (FN + TP)
    Specificity = TN / (FP + TN)

    print ('The Accuracy on valid set is ' + str(Accuracy))
    print ('The Sensitivity on valid set is ' + str(Sensitivity))
    print ('The Specificity on valid set is ' + str(Specificity))

    return (logreg, y_pred_ROC)

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

# Cross validation
def get_cross_validation_and_hyperparameter_tuning(X, y):
    model = LogisticRegression(max_iter=250, penalty='l2', class_weight="balanced")
    solvers = ['newton-cg', 'lbfgs', 'liblinear','sag', 'saga']
    penalty = ['l2']
    c_values = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1]

    grid = dict(solver=solvers,penalty=penalty,C=c_values)
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    logcv = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
    log_cv_param = logcv.fit(X, y)
    y_pred = logcv.predict(X) 

    cm = confusion_matrix(y, y_pred)
    TP, FP, TN, FN = (cm[1][1], cm[0][1], cm[0][0], cm[1][0])
    print (TP, FP, TN, FN)   

    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    Sensitivity = TP / (FN + TP)
    Specificity = TN / (FP + TN)

    print ('The CV Accuracy is ' + str(Accuracy))
    print ('The CV Sensitivity is ' + str(Sensitivity))
    print ('The CV Specificity is ' + str(Specificity))
    print("Best: %f using %s" % (log_cv_param.best_score_, log_cv_param.best_params_))

    return log_cv_param.best_params_['C'], log_cv_param.best_params_['solver']

## Rifampicin
# Train test valid split
X_train, X_test, Rif_train, Rif_test = train_test_split(X, Rif, train_size=0.8, random_state=42)
X_test, X_valid, Rif_test, Rif_valid= train_test_split(X_test, Rif_test, test_size=0.5, random_state=42)

# Cross validation and hyperparameter tuning
L2_Rif, Solver_Rif = get_cross_validation_and_hyperparameter_tuning(X, Rif)

# Fitting logistic regression and assess with confusion matrix
Rif_model, Rif_pred_ROC = get_logistic_regression_and_CM(L2_Rif, Solver_Rif, X_train, X_valid, Rif_train, Rif_valid)

# Search for optimal threshold
Rif_threshold, Rif_fpr, Rif_tpr = get_threshold(Rif_valid, Rif_pred_ROC)

# Prediction and tracking cpu util and time
cpu_usage_percent_list_Rif = []
start_time = time.time()
Rif_test_pred = Rif_model.predict_proba (X_test)
end_time = time.time()
while True:
    cpu_usage_percent = psutil.cpu_percent(interval=1)
    cpu_usage_percent_list_Rif.append(cpu_usage_percent)
    if time.time() - start_time > end_time - start_time:
        break
total_time = end_time - start_time
print(f"Total time taken predicting test set (Rifampicin): {total_time:.2f} seconds")
average_cpu_usage_percent = sum(cpu_usage_percent_list_Rif) / len(cpu_usage_percent_list_Rif)
print(f"Average CPU usage during prediction (Rifampicin): {average_cpu_usage_percent:.2f}%")

# Decide the output with the threshold
Rif_predictions = (Rif_test_pred[:, 1] >= Rif_threshold).astype(int)

# Implement confusion matrix
cm = confusion_matrix(Rif_test, Rif_predictions)
TP, FP, TN, FN = (cm[1][1], cm[0][1], cm[0][0], cm[1][0])
print (TP, FP, TN, FN)   

Accuracy = (TP + TN) / (TP + TN + FP + FN)
Sensitivity = TP / (FN + TP)
Specificity = TN / (FP + TN)

print ('The Accuracy on test set is ' + str(Accuracy))
print ('The Sensitivity on test set is ' + str(Sensitivity))
print ('The Specificity on test set is ' + str(Specificity))

# Plot ROC curve
plt.plot(Rif_fpr, Rif_tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()


## Isoniazid
# Train test valid split
X_train, X_test, Iso_train, Iso_test = train_test_split(X, Iso, train_size=0.8, random_state=42)
X_test, X_valid, Iso_test, Iso_valid= train_test_split(X_test, Iso_test, test_size=0.5, random_state=42)

# Cross validation and hyperparameter tuning
L2_Iso, Solver_Iso = get_cross_validation_and_hyperparameter_tuning(X, Iso)

# Fitting logistic regression and assess with confusion matrix
Iso_model, Iso_pred_ROC = get_logistic_regression_and_CM(L2_Iso, Solver_Iso, X_train, X_valid, Iso_train, Iso_valid)

# Search for optimal threshold
Iso_threshold, Iso_fpr, Iso_tpr = get_threshold(Iso_valid, Iso_pred_ROC)

# Prediction and tracking cpu util and time
cpu_usage_percent_list_Iso = []
start_time = time.time()
Iso_test_pred = Iso_model.predict_proba (X_test)
end_time = time.time()
while True:
    cpu_usage_percent = psutil.cpu_percent(interval=1)
    cpu_usage_percent_list_Iso.append(cpu_usage_percent)
    if time.time() - start_time > end_time - start_time:
        break
total_time = end_time - start_time
print(f"Total time taken predicting test set (Isoniazid): {total_time:.2f} seconds")
average_cpu_usage_percent = sum(cpu_usage_percent_list_Iso) / len(cpu_usage_percent_list_Iso)
print(f"Average CPU usage during prediction (Isoniazid): {average_cpu_usage_percent:.2f}%")

# Decide the output with the threshold
Iso_predictions = (Iso_test_pred[:, 1] >= Iso_threshold).astype(int)

# Implement confusion matrix
cm = confusion_matrix(Iso_test, Iso_predictions)
TP, FP, TN, FN = (cm[1][1], cm[0][1], cm[0][0], cm[1][0])
print (TP, FP, TN, FN)   

Accuracy = (TP + TN) / (TP + TN + FP + FN)
Sensitivity = TP / (FN + TP)
Specificity = TN / (FP + TN)

print ('The Accuracy on test set is ' + str(Accuracy))
print ('The Sensitivity on test set is ' + str(Sensitivity))
print ('The Specificity on test set is ' + str(Specificity))

# Plot ROC curve
plt.plot(Iso_fpr, Iso_tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()


## Ethambutol
# Train test valid split
X_train, X_test, Eth_train, Eth_test = train_test_split(X, Eth, train_size=0.8, random_state=42)
X_test, X_valid, Eth_test, Eth_valid= train_test_split(X_test, Eth_test, test_size=0.5, random_state=42)

# Cross validation and hyperparameter tuning
L2_Eth, Solver_Eth = get_cross_validation_and_hyperparameter_tuning(X, Eth)

# Fitting logistic regression and assess with confusion matrix
Eth_model, Eth_pred_ROC = get_logistic_regression_and_CM(L2_Eth, Solver_Eth, X_train, X_valid, Eth_train, Eth_valid)

# Search for optimal threshold
Eth_threshold, Eth_fpr, Eth_tpr = get_threshold(Eth_valid, Eth_pred_ROC)

# Prediction and tracking cpu util and time
cpu_usage_percent_list_Eth = []
start_time = time.time()
Eth_test_pred = Eth_model.predict_proba (X_test)
end_time = time.time()
while True:
    cpu_usage_percent = psutil.cpu_percent(interval=1)
    cpu_usage_percent_list_Eth.append(cpu_usage_percent)
    if time.time() - start_time > end_time - start_time:
        break
total_time = end_time - start_time
print(f"Total time taken predicting test set (Ethambutol): {total_time:.2f} seconds")
average_cpu_usage_percent = sum(cpu_usage_percent_list_Eth) / len(cpu_usage_percent_list_Eth)
print(f"Average CPU usage during prediction (Ethambutol): {average_cpu_usage_percent:.2f}%")

# Decide the output with the threshold
Eth_predictions = (Eth_test_pred[:, 1] >= Eth_threshold).astype(int)

# Implement confusion matrix
cm = confusion_matrix(Eth_test, Eth_predictions)
TP, FP, TN, FN = (cm[1][1], cm[0][1], cm[0][0], cm[1][0])
print (TP, FP, TN, FN)   

Accuracy = (TP + TN) / (TP + TN + FP + FN)
Sensitivity = TP / (FN + TP)
Specificity = TN / (FP + TN)

print ('The Accuracy on test set is ' + str(Accuracy))
print ('The Sensitivity on test set is ' + str(Sensitivity))
print ('The Specificity on test set is ' + str(Specificity))

# Plot ROC curve
plt.plot(Eth_fpr, Eth_tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()


## Pyrazinamide
# Train test valid split
X_train, X_test, Pyr_train, Pyr_test = train_test_split(X, Pyr, train_size=0.8, random_state=42)
X_test, X_valid, Pyr_test, Pyr_valid= train_test_split(X_test, Pyr_test, test_size=0.5, random_state=42)

# Cross validation and hyperparameter tuning
L2_Pyr, Solver_Pyr = get_cross_validation_and_hyperparameter_tuning(X, Pyr)

# Fitting logistic regression and assess with confusion matrix
Pyr_model, Pyr_pred_ROC = get_logistic_regression_and_CM(L2_Pyr, Solver_Pyr, X_train, X_valid, Pyr_train, Pyr_valid)

# Search for optimal threshold
Pyr_threshold, Pyr_fpr, Pyr_tpr = get_threshold(Pyr_valid, Pyr_pred_ROC)

# Prediction and tracking cpu util and time
cpu_usage_percent_list_Pyr = []
start_time = time.time()
Pyr_test_pred = Pyr_model.predict_proba (X_test)
end_time = time.time()
while True:
    cpu_usage_percent = psutil.cpu_percent(interval=1)
    cpu_usage_percent_list_Pyr.append(cpu_usage_percent)
    if time.time() - start_time > end_time - start_time:
        break
total_time = end_time - start_time
print(f"Total time taken predicting test set (Pyrazinamide): {total_time:.2f} seconds")
average_cpu_usage_percent = sum(cpu_usage_percent_list_Pyr) / len(cpu_usage_percent_list_Pyr)
print(f"Average CPU usage during prediction (Pyrazinamide): {average_cpu_usage_percent:.2f}%")

# Decide the output with the threshold
Pyr_predictions = (Pyr_test_pred[:, 1] >= Pyr_threshold).astype(int)

# Implement confusion matrix
cm = confusion_matrix(Pyr_test, Pyr_predictions)
TP, FP, TN, FN = (cm[1][1], cm[0][1], cm[0][0], cm[1][0])
print (TP, FP, TN, FN)   

Accuracy = (TP + TN) / (TP + TN + FP + FN)
Sensitivity = TP / (FN + TP)
Specificity = TN / (FP + TN)

print ('The Accuracy on test set is ' + str(Accuracy))
print ('The Sensitivity on test set is ' + str(Sensitivity))
print ('The Specificity on test set is ' + str(Specificity))

# Plot ROC curve
plt.plot(Pyr_fpr, Pyr_tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

# Compiling the models
models = {"Rifampicin": Rif_model, "Isoniazid": Iso_model, "Ethambutol": Eth_model, "Pyrazinamide": Pyr_model}

# Save the model as a pickle
with open('LR_model.pickle', 'wb') as f:
    pkl.dump(models, f)

