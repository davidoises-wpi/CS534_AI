"""
ML Models to be used and the tuning parameters:
1. Logistic Regression (Week 5)
2. Decision Tree: criterion, cpp_alpha (Week 6)
1. Artificial Nerual Neworks: hidden_layer_sizes, activation
2. Support Vector Machine: C, Kernel
3. K-Nearest Neighbors: n_neighbors, p
"""

import pandas as pd
import os
import sys
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, matthews_corrcoef

PRINT_DEBUG_OUTPUT = 0
PRINT_VERBOSE_OUTPUT = 0
PRINT_WIP_OUTPUT = 1

def load_dataset(file, columns_to_remove):
    # Load the dataset from a csv file
    df = pd.read_csv(file, index_col='UDI')

    # Remove columns that wont be used for the models
    df = df.dropna()
    df.drop(columns=columns_to_remove, inplace=True)

    return df

def preprocess_dataset(raw_dataset):
    # Caterogical data to numerical
    raw_dataset['Type'].replace(['L', 'M', 'H'], [0, 1, 2], inplace=True)

    # Normalizing data
    normalizer = MinMaxScaler().fit(raw_dataset)
    normalized_array = normalizer.transform(raw_dataset)

    # Covnert back to dataframe format
    processed_dataset = pd.DataFrame(normalized_array, columns=list(raw_dataset.columns))

    return processed_dataset

def resample_dataset(unbalanced_dataset):

    if PRINT_VERBOSE_OUTPUT:

        # Divide the dataset in terms of its classification
        faulted_machines_data = unbalanced_dataset[unbalanced_dataset['Machine failure'] == 1]
        nonfaulted_machines_data = unbalanced_dataset[unbalanced_dataset['Machine failure'] == 0]

        # Print the number of samples before resamping
        print("Faulted machines data before resampling:")
        print("Rows: " + str(faulted_machines_data.shape[0]) + ", Columns: " + str(faulted_machines_data.shape[1]))
        print("Non-faulted machines data before resampling:")
        print("Rows: " + str(nonfaulted_machines_data.shape[0]) + ", Columns: " + str(nonfaulted_machines_data.shape[1]))


    # Sampling strategy = 1.0 means the same amount of fulted and non-faulted samples will be present
    # in the resampled dataset
    rus = RandomUnderSampler(sampling_strategy=1.0)

    # Resample using the machine failure column as the classifier
    balanced_dataset, y_res = rus.fit_resample(unbalanced_dataset, unbalanced_dataset['Machine failure'])

    if PRINT_VERBOSE_OUTPUT:

        # Divide the dataset in terms of its classification
        faulted_machines_data = balanced_dataset[balanced_dataset['Machine failure'] == 1]
        nonfaulted_machines_data = balanced_dataset[balanced_dataset['Machine failure'] == 0]

        # Print the number of samples before resamping
        print("Faulted machines data after resampling:")
        print("Rows: " + str(faulted_machines_data.shape[0]) + ", Columns: " + str(faulted_machines_data.shape[1]))
        print("Non-faulted machines data after resampling:")
        print("Rows: " + str(nonfaulted_machines_data.shape[0]) + ", Columns: " + str(nonfaulted_machines_data.shape[1]))

    return balanced_dataset

def evaluate_estimator_with_matthews_corrcoef(estimator, X, y):
    predictions = estimator.predict(X)
    bin_predictions = [1 if x >= 0.5 else 0 for x in predictions]

    score = matthews_corrcoef(y, bin_predictions)

    return score

def main():

    # Full filename to avoid issues opening file
    input_dataset_file = os.path.join(sys.path[0], "ai4i2020.csv")

    # Loading Predictive Maintainance Dataset and removing undesired fields
    raw_pmd_df = load_dataset(input_dataset_file, ['Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'])

    if PRINT_DEBUG_OUTPUT:
        print("Dataset with raw values")
        print(raw_pmd_df.to_string())

    #Categorical data transformation and normalization
    normd_pmd_df = preprocess_dataset(raw_pmd_df)    

    if PRINT_DEBUG_OUTPUT:
        print("Dataset with categorical data transformation and normalization")
        print(normd_pmd_df.to_string())

    """ Balancing dataset, undersampling non-faulted machines data"""
    pmd_bal_df = resample_dataset(normd_pmd_df)

    if PRINT_DEBUG_OUTPUT:
        print("Dataset after resampling")
        print(pmd_bal_df.to_string())

    """ Training the model"""

    X = pmd_bal_df.drop(columns=['Machine failure'])
    y = pmd_bal_df['Machine failure']

    # Generate training and test datasets, 30% will belong to test dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

    """Logistic Regresion"""
    # clf = LogisticRegression(max_iter=500).fit(X_train, y_train)

    # predictions = clf.predict(X_test)
    # bin_predictions = [1 if x >= 0.5 else 0 for x in predictions]

    # print(accuracy_score(y_test, bin_predictions))
    # print(confusion_matrix(y_test, bin_predictions))
    # print(matthews_corrcoef(y_test, bin_predictions))

    """Decission Tree"""

    # This can be used to understand the ranges of ccp_alphas. In this case it seemed the maximum was 0.2
    # path = clf.cost_complexity_pruning_path(X_train, y_train)
    # ccp_alphas, impurities = path.ccp_alphas, path.impurities

    # Parameters to be tuned in decision tree model
    param_dict = {
        "criterion":['gini', 'entropy', 'log_loss'],
        "ccp_alpha":[x / 1000.0 for x in range(0, 200, 1)]
    }

    # Grid search to try all posibilities based on 5 cross fold validation results
    grid = GridSearchCV(DecisionTreeClassifier(),
                 param_grid=param_dict,
                 cv=5,
                 scoring=evaluate_estimator_with_matthews_corrcoef)
    grid.fit(X_train, y_train)

    # Get the trained model with the best hyperparameters
    clf = grid.best_estimator_
    print(grid.best_score_)

    # predictions = clf.predict(X_test)
    # bin_predictions = [1 if x >= 0.5 else 0 for x in predictions]


    res = cross_val_score(clf, X_train, y_train, cv=5, scoring=evaluate_estimator_with_matthews_corrcoef)
    print(res)

    # print(accuracy_score(y_test, bin_predictions))
    # print(confusion_matrix(y_test, bin_predictions))
    # print(matthews_corrcoef(y_test, bin_predictions))


if __name__ == "__main__":
    main()