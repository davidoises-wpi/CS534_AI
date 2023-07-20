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

PRINT_DEBUG_OUTPUT = 0
PRINT_VERBOSE_OUTPUT = 1
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

""" Categorical data transformation (and normalizing?) """

pmd_df['Type'].replace(['L', 'M', 'H'], [0, 1, 2], inplace=True)

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


rus = RandomUnderSampler(sampling_strategy=1.0)
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