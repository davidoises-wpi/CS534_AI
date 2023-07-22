
import pandas as pd
import os
import sys
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import matthews_corrcoef

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

def train_logistic_regression(X,y):
    clf = LogisticRegression(max_iter=500).fit(X, y)
    score = cross_val_score(clf, X, y, cv=5, scoring=evaluate_estimator_with_matthews_corrcoef)

    return clf, score

def train_decision_tree(X,y):
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
    grid.fit(X, y)

    clf = grid.best_estimator_
    score = cross_val_score(clf, X, y, cv=5, scoring=evaluate_estimator_with_matthews_corrcoef)

    return clf, score, grid.best_params_

def train_kneighnors(X,y):

    # Parameters to be tuned in K nearest neighbor model
    param_dict = {
        "n_neighbors":range(3,21,2),
        "p":[1,2]
    }

    # Grid search to try all posibilities based on 5 cross fold validation results
    grid = GridSearchCV(KNeighborsClassifier(),
                 param_grid=param_dict,
                 cv=5,
                 scoring=evaluate_estimator_with_matthews_corrcoef)
    grid.fit(X, y)

    clf = grid.best_estimator_
    score = cross_val_score(clf, X, y, cv=5, scoring=evaluate_estimator_with_matthews_corrcoef)

    return clf, score, grid.best_params_

def train_svm(X,y):

    # Parameters to be tuned in K nearest neighbor model
    param_dict = {
        "kernel":["linear", "poly", "rbf", "sigmoid"],
        "C":[0.1, 1, 10, 100, 1000]
    }

    # Grid search to try all posibilities based on 5 cross fold validation results
    grid = GridSearchCV(SVC(),
                 param_grid=param_dict,
                 cv=5,
                 scoring=evaluate_estimator_with_matthews_corrcoef)
    grid.fit(X, y)

    clf = grid.best_estimator_
    score = cross_val_score(clf, X, y, cv=5, scoring=evaluate_estimator_with_matthews_corrcoef)

    return clf, score, grid.best_params_

def train_ann(X,y):
    # Parameters to be tuned in ANN model
    param_dict = {
        "activation":["identity", "logistic", "tanh", "relu"],
        "hidden_layer_sizes":[(4, 2), (3, 3), (8, 4), (12, 6, 3), (12, 8, 3)],
        "learning_rate_init":[0.01, 0.1, 0.2],
        "solver":["sgd"],
        "max_iter":[1000],
    }

    # Grid search to try all posibilities based on 5 cross fold validation results
    grid = GridSearchCV(MLPClassifier(),
                 param_grid=param_dict,
                 cv=5,
                 scoring=evaluate_estimator_with_matthews_corrcoef)
    grid.fit(X, y)

    clf = grid.best_estimator_
    score = cross_val_score(clf, X, y, cv=5, scoring=evaluate_estimator_with_matthews_corrcoef)

    return clf, score, grid.best_params_

def main():
    """
    ML Models to be used and their respective tuning hyperparameters:
    1. Logistic Regression: NA (Week 5)
    2. Decision Tree: criterion, cpp_alpha (Week 6)
    3. K-Nearest Neighbors: n_neighbors, p (Week 6)
    4. Support Vector Machine: C, Kernel (Week 6)
    5. Artificial Nerual Neworks: hidden_layer_sizes, activation
    """

    # Full filename to avoid issues opening file
    input_dataset_file = os.path.join(sys.path[0], "ai4i2020.csv")

    # Loading Predictive Maintainance Dataset and removing undesired fields
    raw_pmd_df = load_dataset(input_dataset_file, ['Product ID', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF'])

    if PRINT_VERBOSE_OUTPUT:
        print("Dataset with raw values")
        print(raw_pmd_df.to_string())

    #Categorical data transformation and normalization
    normd_pmd_df = preprocess_dataset(raw_pmd_df)    

    if PRINT_VERBOSE_OUTPUT:
        print("Dataset with categorical data transformation and normalization")
        print(normd_pmd_df.to_string())

    """ Balancing dataset, undersampling non-faulted machines data"""
    pmd_bal_df = resample_dataset(normd_pmd_df)

    if PRINT_VERBOSE_OUTPUT:
        print("Dataset after resampling")
        print(pmd_bal_df.to_string())

    """ Training the models"""

    X = pmd_bal_df.drop(columns=['Machine failure'])
    y = pmd_bal_df['Machine failure']

    # Generate training and test datasets, 30% will belong to test dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

    """Logistic Regresion"""
    logistic_regression_estimator, logistic_regression_mcc_score = train_logistic_regression(X_train, y_train)
    logistic_regression_mcc_score_avg = sum(logistic_regression_mcc_score)/len(logistic_regression_mcc_score)

    """Decission Tree"""
    decision_tree_estimator, decision_tree_mcc_score, decision_tree_best_params = train_decision_tree(X_train, y_train)
    decision_tree_mcc_score_avg = sum(decision_tree_mcc_score)/len(decision_tree_mcc_score)

    """K-Nearest Neighbors"""
    kneighbors_estimator, kneighbors_mcc_score, kneighbors_best_params = train_kneighnors(X_train, y_train)
    kneighbors_mcc_score_avg = sum(kneighbors_mcc_score)/len(kneighbors_mcc_score)

    """K-Nearest Neighbors"""
    svm_estimator, svm_mcc_score, svm_best_params = train_svm(X_train, y_train)
    svm_mcc_score_avg = sum(svm_mcc_score)/len(svm_mcc_score)

    """Artificial Neural Network"""
    ann_estimator, ann_mcc_score, ann_best_params = train_ann(X_train, y_train)
    ann_mcc_score_avg = sum(ann_mcc_score)/len(ann_mcc_score)
        
    # Generat a dataframe with the training results. Pandas dataframe makes it easier to print in a table form
    data = {
        'Best Set Of Parameters': ["NA", decision_tree_best_params, kneighbors_best_params, svm_best_params, ann_best_params],
        'MCC score on training set': [logistic_regression_mcc_score_avg, decision_tree_mcc_score_avg, kneighbors_mcc_score_avg, svm_mcc_score_avg, ann_mcc_score_avg]
    }
    algorithm_names = ["Logistic Regression", "Decision Tree", "K Nearest Neighbors", "SVM", "ANN"]
    training_results = pd.DataFrame(data, index=algorithm_names)

    if PRINT_DEBUG_OUTPUT:
        print()
        print(training_results.to_string())
        print()

    # Now used the trained classifiers with the tests datasets
    logistic_regression_mcc_score = cross_val_score(logistic_regression_estimator, X_test, y_test, cv=5, scoring=evaluate_estimator_with_matthews_corrcoef)
    logistic_regression_mcc_score_avg = sum(logistic_regression_mcc_score)/len(logistic_regression_mcc_score)

    decision_tree_mcc_score = cross_val_score(decision_tree_estimator, X_test, y_test, cv=5, scoring=evaluate_estimator_with_matthews_corrcoef)
    decision_tree_mcc_score_avg = sum(decision_tree_mcc_score)/len(decision_tree_mcc_score)

    kneighbors_mcc_score = cross_val_score(kneighbors_estimator, X_test, y_test, cv=5, scoring=evaluate_estimator_with_matthews_corrcoef)
    kneighbors_mcc_score_avg = sum(kneighbors_mcc_score)/len(kneighbors_mcc_score)

    svm_mcc_score = cross_val_score(svm_estimator, X_test, y_test, cv=5, scoring=evaluate_estimator_with_matthews_corrcoef)
    svm_mcc_score_avg = sum(svm_mcc_score)/len(svm_mcc_score)

    ann_mcc_score = cross_val_score(ann_estimator, X_test, y_test, cv=5, scoring=evaluate_estimator_with_matthews_corrcoef)
    ann_mcc_score_avg = sum(ann_mcc_score)/len(ann_mcc_score)

    # Generat a dataframe with the training results. Pandas dataframe makes it easier to print in a table form
    data = {
        'Best Set Of Parameters': ["NA", decision_tree_best_params, kneighbors_best_params, svm_best_params, ann_best_params],
        'MCC score on test set': [logistic_regression_mcc_score_avg, decision_tree_mcc_score_avg, kneighbors_mcc_score_avg, svm_mcc_score_avg, ann_mcc_score_avg]
    }
    algorithm_names = ["Logistic Regression", "Decision Tree", "K Nearest Neighbors", "SVM", "ANN"]
    test_results = pd.DataFrame(data, index=algorithm_names)

    if PRINT_DEBUG_OUTPUT:
        print()
        print(test_results.to_string())
        print()

    # Find the best algorithm best on the max score gotten on the test set
    best_algorithm_index = pd.Series([logistic_regression_mcc_score_avg, decision_tree_mcc_score_avg, kneighbors_mcc_score_avg, svm_mcc_score_avg, ann_mcc_score_avg]).idxmax()
    print()
    print("The best algorithm to be used for this problem is the " + algorithm_names[best_algorithm_index])


if __name__ == "__main__":
    main()