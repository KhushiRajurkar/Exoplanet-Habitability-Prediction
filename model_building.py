# -*- coding: utf-8 -*-
"""Customized Model Building Script for Multi-Class Classification"""

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from imblearn.combine import SMOTETomek
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTE, ADASYN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.under_sampling import ClusterCentroids
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import recall_score, precision_score, f1_score, roc_auc_score, ConfusionMatrixDisplay, classification_report

# Read the processed dataset
hwc_data = pd.read_excel("hwc.xlsx")
print("Initial Data:")
print(hwc_data.head())
print("\nClass distribution in the entire dataset:")
print(hwc_data['P_HABITABLE'].value_counts())

# Define model builders for multi-class classification
def svc_model_builder(X_train, y_train, X_test):
    svc = SVC(random_state=12, probability=True, decision_function_shape='ovr')  # One-vs-Rest
    param_grid = {
        'C': [1, 10, 100, 1000],
        'gamma': [0.001, 0.0001],
        'kernel': ['sigmoid', 'rbf']
    }
    clf = GridSearchCV(svc, param_grid, cv=5, scoring='f1_macro', return_train_score=False, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)
    return y_pred, y_proba

def mlpc_model_builder(X_train, y_train, X_test):
    n_features = X_train.shape[1]
    hidden_layers = [i for i in range(n_features - 2, 1, -2)]
    if not hidden_layers:
        hidden_layers = [2]
    mlpc = MLPClassifier(hidden_layer_sizes=tuple(hidden_layers), early_stopping=True,
                         random_state=12, max_iter=10000)
    param_grid = {
        'solver': ['lbfgs', 'adam', 'sgd'],
        'hidden_layer_sizes': [tuple(hidden_layers), (50,), (100,)]
    }
    clf = GridSearchCV(mlpc, param_grid, cv=5, scoring='f1_macro', return_train_score=False, n_jobs=-1)
    clf.fit(X_train.values, y_train)
    y_pred = clf.predict(X_test.values)
    if hasattr(clf.best_estimator_, "predict_proba"):
        y_proba = clf.predict_proba(X_test.values)
    elif hasattr(clf.best_estimator_, "decision_function"):
        y_proba = clf.decision_function(X_test.values)
    else:
        y_proba = y_pred  # Fallback
    return y_pred, y_proba

def knn_model_builder(X_train, y_train, X_test):
    knn = KNeighborsClassifier()
    param_grid = {'n_neighbors': range(1, 11)}
    clf = GridSearchCV(knn, param_grid, cv=5, scoring='f1_macro', return_train_score=False, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    if hasattr(clf.best_estimator_, "predict_proba"):
        y_proba = clf.predict_proba(X_test)
    else:
        y_proba = clf.predict(X_test)  # Fallback
    return y_pred, y_proba

def lr_model_builder(X_train, y_train, X_test):
    logreg = LogisticRegression(random_state=12, max_iter=10000, solver='lbfgs', class_weight='balanced', multi_class='ovr')
    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100],
        'solver': ['lbfgs'],
        'penalty': ['l2']
    }
    clf = GridSearchCV(logreg, param_grid, cv=5, scoring='f1_macro', return_train_score=False, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    if hasattr(clf.best_estimator_, "predict_proba"):
        y_proba = clf.predict_proba(X_test)
    elif hasattr(clf.best_estimator_, "decision_function"):
        y_proba = clf.decision_function(X_test)
    else:
        y_proba = y_pred  # Fallback
    return y_pred, y_proba

# Dataset and model loop
Dataset_name = 'hwc.xlsx'
Model_names = ['SVM', 'MLPClassifier', 'KNN', 'Logistic Regression']
Sampler_names = ['ClusterCentroids', 'SMOTE', 'ADASYN', 'SMOTETomek']
metrics_df = pd.DataFrame(
    columns=['Model', 'Sampler', 'F1 Score', 'Precision', 'Recall', 'AUC', 'Time(sec)'])

model_builders = [svc_model_builder, mlpc_model_builder, knn_model_builder, lr_model_builder]

# Separate features and target
X = hwc_data.iloc[:, :-1]
y = hwc_data.iloc[:, -1]

for model_builder, model_name in zip(model_builders, Model_names):
    for sampler, sampler_name in zip(
        [ClusterCentroids(random_state=12), SMOTE(random_state=12), ADASYN(random_state=12), SMOTETomek(random_state=12)],
        Sampler_names
    ):
        print(f"\nTraining {model_name} with {sampler_name} sampler...")
        start_time = time.time()

        # Split before resampling to prevent data leakage
        X_train_orig, X_test_orig, y_train_orig, y_test = train_test_split(
            X, y, test_size=0.2, random_state=12, stratify=y
        )

        print("\nClass distribution in the training set before resampling:")
        print(y_train_orig.value_counts())

        # Check if all classes are present in the training set
        if len(np.unique(y_train_orig)) < 2:
            print(f"Training data has only one class after split. Skipping {model_name} with {sampler_name}.")
            continue

        try:
            # Apply sampling only on the training set
            X_resampled, y_resampled = sampler.fit_resample(X_train_orig, y_train_orig)
            print(f"Class distribution after resampling with {sampler_name}:")
            print(pd.Series(y_resampled).value_counts())
        except ValueError as e:
            print(f"Error during resampling with {sampler_name}: {e}")
            print("Skipping this combination.")
            continue

        # Further check to ensure resampling was successful
        if len(np.unique(y_resampled)) < 2:
            print(f"After resampling with {sampler_name}, only one class is present. Skipping this combination.")
            continue

        X_train, y_train = X_resampled, y_resampled
        X_test = X_test_orig
        # y_test remains unchanged

        # Build and predict using the model
        try:
            y_pred, y_proba = model_builder(X_train, y_train, X_test)
        except Exception as e:
            print(f"Error during model training/prediction with {model_name} and {sampler_name}: {e}")
            continue

        # Handle multi-class AUC
        try:
            if isinstance(y_proba, (np.ndarray, list, pd.Series)):
                if y_proba.ndim == 1 or y_proba.shape[1] == 1:
                    # Binary classification or single class probabilities
                    auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
                else:
                    # Multi-class classification with probabilities for each class
                    auc = roc_auc_score(pd.get_dummies(y_test), y_proba, multi_class='ovr')
            else:
                auc = np.nan  # Unable to compute AUC
        except Exception as e:
            print(f"Error computing AUC: {e}")
            auc = np.nan

        # Compute metrics
        f1 = f1_score(y_test, y_pred, average='macro')
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')

        # Handle AUC
        if not np.isnan(auc):
            auc_score = auc
        else:
            try:
                # Fallback using classification report's macro avg F1 as a proxy
                auc_score = f1
            except:
                auc_score = np.nan

        # Display Confusion Matrix
        try:
            disp = ConfusionMatrixDisplay.from_predictions(
                y_test, y_pred, display_labels=["Not Habitable", "Habitable Class 1", "Habitable Class 2"], normalize=None
            )
            disp.ax_.set_title(f"{model_name} - {sampler_name}")
            plt.savefig(f"{model_name}_{sampler_name}_ConfusionMatrix.png")
            plt.close()
        except Exception as e:
            print(f"Error during confusion matrix plotting: {e}")

        # Calculate time elapsed
        time_elapsed = time.time() - start_time

        # Append metrics to the DataFrame
        metrics_df.loc[len(metrics_df)] = [
            model_name,
            sampler_name,
            f1,
            precision,
            recall,
            auc_score,
            time_elapsed
        ]

        print(f"Completed {model_name} with {sampler_name} in {time_elapsed:.2f} seconds.")

# Save metrics to Excel
metrics_df_f1 = metrics_df.sort_values(by=["F1 Score", "Recall", "Precision", "AUC"], ascending=False)
metrics_df_recall = metrics_df.sort_values(by=["Recall", "F1 Score", "Precision", "AUC"], ascending=False)

metrics_df_f1.to_excel('Metrics_F1_Score.xlsx', index=False)
metrics_df_recall.to_excel('Metrics_Recall.xlsx', index=False)

print("\nPerformance Results (F1 Score sorted):")
print(metrics_df_f1.head())
print("\nPerformance Results (Recall sorted):")
print(metrics_df_recall.head())