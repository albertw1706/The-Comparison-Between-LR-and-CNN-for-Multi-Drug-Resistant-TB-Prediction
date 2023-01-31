import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve

DF = pd.read_csv("Modeltrial_Scale_100.csv")

DF ['Rifampicin'] = DF ['Rifampicin'].replace({'S': 0, 'R': 1}, regex=True)
DF ['Isoniazid'] = DF ['Isoniazid'].replace({'S': 0, 'R': 1}, regex=True)
DF ['Ethambutol'] = DF ['Ethambutol'].replace({'S': 0, 'R': 1}, regex=True)
DF ['Pyrazinamide'] = DF ['Pyrazinamide'].replace({'S': 0, 'R': 1}, regex=True)
print (DF)

data = DF.columns.values.tolist()
print (data)

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

def get_logistic_regression_and_CM(L2_param, X_train, X_valid, y_train, y_valid):
    logreg = LogisticRegression(max_iter=250, C= L2_param, penalty='l2', solver='saga', random_state=0)
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_valid)
    y_pred_ROC = logreg.predict_proba(X_valid)[:, 1]
    print('Accuracy of logistic regression classifier on valid set: {:.2f}'.format(logreg.score(X_valid, y_valid)))

    cm = confusion_matrix(y_valid, y_pred)
    TP, FP, TN, FN = (cm[1][1], cm[0][1], cm[0][0], cm[1][0])
    print (TP, FP, TN, FN)   

    Accuracy = (TP + TN) / (TP + TN + FP + TP)
    Sensitivity = TP / (FN + TP)
    Specificity = TN / (FP + TN)

    print ('The Accuracy on valid set is ' + str(Accuracy))
    print ('The Sensitivity on valid set is ' + str(Sensitivity))
    print ('The Specificity on valid set is ' + str(Specificity))

    return (logreg, y_pred_ROC)

# Hyperparameter Tuning L2
def hyperparameter_L2_and_solver(X_train, y_train):
    kf = KFold(n_splits=10, shuffle=True, random_state=0)
    param_grid = {"alpha": np.arange(0.00001, 1, 10), "solver": ["sag", "saga"]}
    ridge = Ridge()
    ridge_cv = GridSearchCV(ridge, param_grid, cv=kf)
    ridge_cv.fit(X_train, y_train)
    print(ridge_cv.best_params_, ridge_cv.best_score_)
    return ridge_cv.best_params_["solver"]

# ROC and Index of Union / Youden for Threshold
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
def get_cross_validation(X, y):
    logr = LogisticRegressionCV(Cs= np.arange(0.00001, 1, 10), cv=10, penalty='l2', solver='saga', max_iter=250, random_state=0)
    logr.fit(X, y)
    y_pred = logr.predict(X)
    best_C = logr.C_[0]

    cm = confusion_matrix(y, y_pred)
    TP, FP, TN, FN = (cm[1][1], cm[0][1], cm[0][0], cm[1][0])
    print (TP, FP, TN, FN)   

    Accuracy = (TP + TN) / (TP + TN + FP + TP)
    Sensitivity = TP / (FN + TP)
    Specificity = TN / (FP + TN)

    print ('The CV Accuracy on valid set is ' + str(Accuracy))
    print ('The CV Sensitivity on valid set is ' + str(Sensitivity))
    print ('The CV Specificity on valid set is ' + str(Specificity))
    print (best_C)

    return best_C

## Rifampicin
X_train, X_test, Rif_train, Rif_test = train_test_split(X, Rif, train_size=0.8, random_state=42)
X_test, X_valid, Rif_test, Rif_valid= train_test_split(X_test, Rif_test, test_size=0.5, random_state=42)
Solver_Rif = hyperparameter_L2_and_solver(X_train, Rif_train)
L2_Rif = get_cross_validation(X, Rif)
Rif_model, Rif_pred_ROC = get_logistic_regression_and_CM(L2_Rif, X_train, X_valid, Rif_train, Rif_valid)
threshold, Rif_fpr, Rif_tpr = get_threshold(Rif_valid, Rif_pred_ROC)

Rif_test_pred = Rif_model.predict_proba (X_test)
predictions = (Rif_test_pred[:, 1] >= threshold).astype(int)

cm = confusion_matrix(Rif_test, predictions)
TP, FP, TN, FN = (cm[1][1], cm[0][1], cm[0][0], cm[1][0])
print (TP, FP, TN, FN)   

Accuracy = (TP + TN) / (TP + TN + FP + TP)
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
X_train, X_test, Iso_train, Iso_test = train_test_split(X, Iso, train_size=0.8, random_state=42)
X_test, X_valid, Iso_test, Iso_valid= train_test_split(X_test, Iso_test, test_size=0.5, random_state=42)
Solver_Iso = hyperparameter_L2_and_solver(X_train, Iso_train)
L2_Iso = get_cross_validation(X, Iso)
Iso_model, Iso_pred_ROC = get_logistic_regression_and_CM(L2_Iso, X_train, X_valid, Iso_train, Iso_valid)
threshold, Iso_fpr, Iso_tpr = get_threshold(Iso_valid, Iso_pred_ROC)

Iso_test_pred = Iso_model.predict_proba (X_test)
predictions = (Iso_test_pred[:, 1] >= threshold).astype(int)

cm = confusion_matrix(Iso_test, Iso_test_pred)
TP, FP, TN, FN = (cm[1][1], cm[0][1], cm[0][0], cm[1][0])
print (TP, FP, TN, FN)   

Accuracy = (TP + TN) / (TP + TN + FP + TP)
Sensitivity = TP / (FN + TP)
Specificity = TN / (FP + TN)

print ('The Accuracy on valid set is ' + str(Accuracy))
print ('The Sensitivity on valid set is ' + str(Sensitivity))
print ('The Specificity on valid set is ' + str(Specificity))

# Plot ROC curve
plt.plot(Iso_fpr, Iso_tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()


## Ethambutol
X_train, X_test, Eth_train, Eth_test = train_test_split(X, Eth, train_size=0.8, random_state=42)
X_test, X_valid, Eth_test, Eth_valid= train_test_split(X_test, Eth_test, test_size=0.5, random_state=42)
Solver_Eth = hyperparameter_L2_and_solver(X_train, Eth_train)
L2_Eth = get_cross_validation(X, Eth)
Eth_model, Eth_pred_ROC = get_logistic_regression_and_CM(L2_Eth, X_train, X_valid, Eth_train, Eth_valid)
threshold, Eth_fpr, Eth_tpr = get_threshold(Eth_valid, Eth_pred_ROC)

Eth_test_pred = Eth_model.predict_proba (X_test)
predictions = (Eth_test_pred[:, 1] >= threshold).astype(int)

cm = confusion_matrix(Eth_test, Eth_test_pred)
TP, FP, TN, FN = (cm[1][1], cm[0][1], cm[0][0], cm[1][0])
print (TP, FP, TN, FN)   

Accuracy = (TP + TN) / (TP + TN + FP + TP)
Sensitivity = TP / (FN + TP)
Specificity = TN / (FP + TN)

print ('The Accuracy on valid set is ' + str(Accuracy))
print ('The Sensitivity on valid set is ' + str(Sensitivity))
print ('The Specificity on valid set is ' + str(Specificity))

# Plot ROC curve
plt.plot(Eth_fpr, Eth_tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()


## Pyrazinamide
X_train, X_test, Pyr_train, Pyr_test = train_test_split(X, Pyr, train_size=0.8, random_state=42)
X_test, X_valid, Pyr_test, Pyr_valid= train_test_split(X_test, Pyr_test, test_size=0.5, random_state=42)
Solver_Pyr = hyperparameter_L2_and_solver(X_train, Pyr_train)
L2_Pyr = get_cross_validation(X, Pyr)
Pyr_model, Pyr_pred_ROC = get_logistic_regression_and_CM(L2_Pyr, X_train, X_valid, Pyr_train, Pyr_valid)
threshold, Pyr_fpr, Pyr_tpr = get_threshold(Pyr_valid, Pyr_pred_ROC)

Pyr_test_pred = Pyr_model.predict_proba (X_test)
predictions = (Pyr_test_pred[:, 1] >= threshold).astype(int)

cm = confusion_matrix(Pyr_test, Pyr_test_pred)
TP, FP, TN, FN = (cm[1][1], cm[0][1], cm[0][0], cm[1][0])
print (TP, FP, TN, FN)   

Accuracy = (TP + TN) / (TP + TN + FP + TP)
Sensitivity = TP / (FN + TP)
Specificity = TN / (FP + TN)

print ('The Accuracy on valid set is ' + str(Accuracy))
print ('The Sensitivity on valid set is ' + str(Sensitivity))
print ('The Specificity on valid set is ' + str(Specificity))

# Plot ROC curve
plt.plot(Pyr_fpr, Pyr_tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()

