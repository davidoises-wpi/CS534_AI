"""
ML Models to be used and the tuning parameters:
1. Logistic Regression (Week 5)
1. Artificial Nerual Neworks: hidden_layer_sizes, activation
2. Support Vector Machine: C, Kernel
3. K-Nearest Neighbors: n_neighbors, p
4. Decision Tree: criterion, cpp_alpha
"""

import pandas as pd
import os
import sys
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler

from sklearn.metrics import accuracy_score, confusion_matrix, matthews_corrcoef

PRINT_DEBUG_OUTPUT = 0
PRINT_VERBOSE_OUTPUT = 0
PRINT_WIP_OUTPUT = 1

""" Loading dataset """
# Full filename to avoid issues opening file
input_dataset_file = os.path.join(sys.path[0], "ai4i2020.csv")

# Load the Predictive Maintainance Dataset
pmd_df = pd.read_csv(input_dataset_file, index_col='UDI')

# Remove columns that wont be used for the models
pmd_df = pmd_df.dropna()
pmd_df.drop(columns=['Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'], inplace=True)

if PRINT_DEBUG_OUTPUT:
    print("Dataset with raw values")
    print(pmd_df.to_string())

""" Categorical data transformation and normalization """

pmd_df['Type'].replace(['L', 'M', 'H'], [0, 1, 2], inplace=True)

normalizer = MinMaxScaler().fit(pmd_df)
normalized_array = normalizer.transform(pmd_df)
pmd_df = pd.DataFrame(normalized_array, columns=list(pmd_df.columns))

if PRINT_DEBUG_OUTPUT:
    print("Dataset with categorical data transformation")
    print(pmd_df.to_string())

""" Balancing dataset, undersampling non-faulted machines data"""

if PRINT_VERBOSE_OUTPUT:

    faulted_machines_data = pmd_df[pmd_df['Machine failure'] == 1]
    nonfaulted_machines_data = pmd_df[pmd_df['Machine failure'] == 0]

    print("Faulted machines data before resampling:")
    print("Rows: " + str(faulted_machines_data.shape[0]) + ", Columns: " + str(faulted_machines_data.shape[1]))

    print("Non-faulted machines data before resampling:")
    print("Rows: " + str(nonfaulted_machines_data.shape[0]) + ", Columns: " + str(nonfaulted_machines_data.shape[1]))


# Sampling strategy = 1.0 means the same amount of fulted and non-faulted samples will be present
# in the resampled dataset
rus = RandomUnderSampler(sampling_strategy=1.0)

# Resample using the machine failure column as the classifier
pmd_bal_df, y_res = rus.fit_resample(pmd_df, pmd_df['Machine failure'])

if PRINT_VERBOSE_OUTPUT:

    faulted_machines_data = pmd_bal_df[pmd_bal_df['Machine failure'] == 1]
    nonfaulted_machines_data = pmd_bal_df[pmd_bal_df['Machine failure'] == 0]

    print("Faulted machines data after resampling:")
    print("Rows: " + str(faulted_machines_data.shape[0]) + ", Columns: " + str(faulted_machines_data.shape[1]))

    print("Non-faulted machines data after resampling:")
    print("Rows: " + str(nonfaulted_machines_data.shape[0]) + ", Columns: " + str(nonfaulted_machines_data.shape[1]))

if PRINT_DEBUG_OUTPUT:
    print("Dataset after resampling")
    print(pmd_bal_df.to_string())

""" Training the model"""

X = pmd_bal_df.drop(columns=['Machine failure'])
y = pmd_bal_df['Machine failure']

# Generate training and test datasets, 30% will belong to test dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

clf = LogisticRegression(max_iter=500).fit(X_train, y_train)
# score = clf.score(X_test, y_test)
# print(score)

predictions = clf.predict(X_test)
bin_predictions = [1 if x >= 0.5 else 0 for x in predictions]

print(accuracy_score(y_test, bin_predictions))
print(confusion_matrix(y_test, bin_predictions))
print(matthews_corrcoef(y_test, bin_predictions))

# scores = cross_val_score(clf, X_train, y_train, cv=5)
# print(scores)