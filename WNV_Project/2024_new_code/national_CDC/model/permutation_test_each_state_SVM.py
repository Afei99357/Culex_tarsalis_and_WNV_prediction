## permutation test for each state

## import the required libraries for permutation test
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn import ensemble, metrics
import pandas as pd


# Load the dataset into a Pandas DataFrame
data = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/WNV/CDC_data/"
                   "human/cdc_human_1999_to_2023/WNV_human_and_non_human_yearly_climate_demographic_bird.csv", index_col=0)

## remove any space and comma in Population column
data["Population"] = data["Population"].str.replace(",", "").str.strip()

## convert Population column to numeric
data["Population"] = pd.to_numeric(data["Population"], errors='coerce')

## build a function to use random forest to train and test the data

# Function adapted for a time-based split and permutation test
def permutation_test_with_time_split(data, target_column, state, n_permutations=1000):

    # Split data based on the year (pre-2018 for training, 2018 onwards for testing)
    train_data = data[data['Year'] < 2018]

    y_train = train_data[target_column]
    # Select columns after the "Date" column as predictors
    X_train = train_data.iloc[:, date_index + 1:]

    test_data = data[data['Year'] >= 2018]
    y_test = test_data[target_column]
    # Select columns after the "Date" column as predictors
    X_test = test_data.iloc[:, date_index + 1:]

    ## if the train and test data are empty, return None
    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
        print("state: ", state, " has no data for training and testing.")
        return None

    ## normalize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train the model on the original training data
    clf = SVR(epsilon=.3, gamma=0.002, kernel="rbf", C=100)

    clf.fit(X_train, y_train)

    y_predict = clf.predict(X_test)

    observed_q2 = metrics.r2_score(y_test, y_predict)

    # Compute Q2 scores for permuted targets within the test set
    permuted_q2_scores = []
    for _ in range(n_permutations):
        # Permute the target variable in the test set only
        y_test_permuted = np.random.permutation(y_test)
        permuted_q2 = metrics.r2_score(y_test_permuted, y_predict)
        permuted_q2_scores.append(permuted_q2)


    ## start a plot
    plt.figure(figsize=(10, 6))
    ## plot the permutaed_q2_scores, and adding 95% significance interval
    plt.hist(permuted_q2_scores, bins=20, color='lightblue', edgecolor='black')

    ## add a vertical line for the observed_q2
    plt.axvline(observed_q2, color='red', linestyle='dashed', linewidth=2)

    ## add a vertical line for the 95% significance interval
    plt.axvline(np.percentile(permuted_q2_scores, 97.5), color='purple', linestyle='dashed', linewidth=2)

    plt.xlabel('Q2 Score')
    plt.ylabel('Frequency')
    plt.title('Permutation Test for Q2 Score in state: ' + data['State'].iloc[0])
    plt.savefig('/Users/ericliao/Desktop/WNV_project_files/WNV/CDC_data/human/result/SVM_each_state/plot/Permutation_test_Q2_score_state_' + data['State'].iloc[0] + '.png')
    plt.close()

    # Determine the p-value
    p_value = np.mean(np.array(permuted_q2_scores) >= observed_q2)
    return observed_q2, p_value

# Perform permutation test for each state
states = data['State'].unique()
results = []
for state in states:
    state_data = data[data['State'] == state]

    ## prting the state
    print("start State: ", state)

    # select the columns after column Date as predictors
    date_index = state_data.columns.get_loc("Date")

    ## drop any nan values in the columns after the Date column
    state_data = state_data.dropna(subset=data.columns[date_index + 1:])

    ## if the state has no data, continue to the next state
    if state_data.shape[0] == 0:
        print("state: ", state, " has no data.")
        continue

    observed_q2, p_value = permutation_test_with_time_split(state_data, 'Neuroinvasive_disease_cases', state, n_permutations=1000)
    results.append({'State': state, 'Observed_q2': observed_q2, 'P_value': p_value})



