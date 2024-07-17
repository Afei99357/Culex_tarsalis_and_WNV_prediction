from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn import ensemble, metrics
import pandas as pd
from sklearn.utils import resample

# Load the dataset into a Pandas DataFrame
data = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/WNV/CDC_data/"
                   "human/cdc_human_1999_to_2023/WNV_human_and_non_human_yearly_climate_demographic_bird.csv", index_col=0)

## remove any space and comma in Population column
data["Population"] = data["Population"].str.replace(",", "").str.strip()

## convert Population column to numeric
data["Population"] = pd.to_numeric(data["Population"], errors='coerce')


## define subsampling function
def balance_classes(data, target_column):
    # Separate majority and minority classes
    data_majority = data[data[target_column] == 0]
    data_minority = data[data[target_column] > 0]

    # Downsample majority class or Upsample minority class
    if len(data_majority) > len(data_minority):
        data_majority_downsampled = resample(data_majority,
                                             replace=False,  # sample without replacement
                                             n_samples=len(data_minority),  # to match minority class
                                             random_state=123)  # reproducible results
        # Combine minority class with downsampled majority class
        data_balanced = pd.concat([data_majority_downsampled, data_minority])
    else:
        data_minority_upsampled = resample(data_minority,
                                           replace=True,  # sample with replacement
                                           n_samples=len(data_majority),  # to match majority class
                                           random_state=123)  # reproducible results
        # Combine majority class with upsampled minority class
        data_balanced = pd.concat([data_minority_upsampled, data_majority])

    return data_balanced

## build a function to use SVM to train and test the data
def svm_model(data, target_column, state):
    ## get only the data for the state
    data_state = data[data["State"] == state]

    # select the columns after column Date as predictors
    date_index = data_state.columns.get_loc("Date")

    # Select columns after the "Date" column as predictors
    data_pred = data_state.iloc[:, date_index+1:]

    # get Year column from data and add it to data_pred
    data_pred['Year'] = data_state['Year']

    ## add target column
    data_pred[target_column] = data_state[target_column]

    # drop nan values
    data_pred = data_pred.dropna()

    ## reset index
    data_pred = data_pred.reset_index(drop=True)

    ## balance the classes
    data_pred = balance_classes(data_pred, target_column)

    ## get the number of rows in the data
    num_rows = data_pred.shape[0]

    ## get the number of rows that are not 0
    non_0_num_rows = data_pred[data_pred[target_column] >= 1].shape[0]

    ### train and test data ####
    train = data_pred[(data_pred["Year"] < 2018)]
    test = data_pred[(data_pred["Year"] >= 2018)]

    # Get labels
    train_labels = train.pop(target_column).values
    test_labels = test.pop(target_column).values

    ## remove time column
    train.pop("Year")
    test.pop("Year")

    ## check if the train and test data are not empty
    if train.shape[0] == 0 or test.shape[0] == 0:
        print("state: ", state, " has no data for training and testing.")
        return None, None, None, None, None

    ## standardize the data
    scaler = StandardScaler()
    train = scaler.fit_transform(train)
    test = scaler.transform(test)

    ######################### SVM ######################################
    ## SVM
    clf = SVR(epsilon=.3, gamma=0.002, kernel="rbf", C=100)

    clf.fit(train, train_labels)

    y_predict = clf.predict(test)

    ## if there is neagtive value in the prediction, change to 0
    y_predict = [0 if i < 0 else i for i in y_predict]

    # ## get the mse, q2 score
    mse = metrics.mean_squared_error(test_labels, y_predict)
    msle = metrics.mean_squared_log_error(test_labels, y_predict)
    Q2 = metrics.r2_score(test_labels, y_predict)

    ## print State, mse, and q2 score
    print("State: ", state, " mse: ", mse, " msle: ", msle, " Q2: ", Q2)

    ## return the mse and q2 score
    return mse, msle, Q2, num_rows, non_0_num_rows

## get unique states
states = data["State"].unique()

## create a dictionary to store the mse and Q2 score for each state
state_mse_msle_q2 = {}

## for each state, get the mse and q2 score
for state in states:
    print("state: ", state)
    mse, msle, q2, num_rows, non_0_num_rows = svm_model(data, "Neuroinvasive_disease_cases", state)
    state_mse_msle_q2[state] = {"mse": mse, "msle": msle, "q2": q2, "num_rows": num_rows, "non_0_num_rows": non_0_num_rows}

## remove the None values from the dictionary
state_mse_msle_q2 = {state: value for state, value in state_mse_msle_q2.items() if value["mse"] is not None}

## create a dictionary for the lower case state names and its initial names
state_name_dict = {"alabama": "AL", "alaska": "AK", "arizona": "AZ", "arkansas": "AR", "california": "CA",
                     "colorado": "CO", "connecticut": "CT", "delaware": "DE", "florida": "FL", "georgia": "GA",
                        "hawaii": "HI", "idaho": "ID", "illinois": "IL", "indiana": "IN", "iowa": "IA", "kansas": "KS",
                        "kentucky": "KY", "louisiana": "LA", "maine": "ME", "maryland": "MD", "massachusetts": "MA",
                        "michigan": "MI", "minnesota": "MN", "mississippi": "MS", "missouri": "MO", "montana": "MT",
                        "nebraska": "NE", "nevada": "NV", "new hampshire": "NH", "new jersey": "NJ", "new mexico": "NM",
                        "new york": "NY", "north carolina": "NC", "north dakota": "ND", "ohio": "OH", "oklahoma": "OK",
                        "oregon": "OR", "pennsylvania": "PA", "rhode island": "RI", "south carolina": "SC",
                        "south dakota": "SD", "tennessee": "TN", "texas": "TX", "utah": "UT", "vermont": "VT",
                        "virginia": "VA", "washington": "WA", "west virginia": "WV", "wisconsin": "WI", "wyoming": "WY"}

## get the state names
state_names = list(state_mse_msle_q2.keys())
mse = [state_mse_msle_q2[state]["mse"] for state in state_names]
msle = [state_mse_msle_q2[state]["msle"] for state in state_names]
q2 = [state_mse_msle_q2[state]["q2"] for state in state_names]
num_rows = [state_mse_msle_q2[state]["num_rows"] for state in state_names]
non_0_num_rows = [state_mse_msle_q2[state]["non_0_num_rows"] for state in state_names]

## get the new x axis labels
state_names = [state_name_dict[state] for state in state_names]

## create a new dataframe
state_mse_msle_q2_df = pd.DataFrame()
state_mse_msle_q2_df["State"] = state_names
state_mse_msle_q2_df["mse"] = mse
state_mse_msle_q2_df["msle"] = msle
state_mse_msle_q2_df["q2"] = q2

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Create a subplot with 2 row and 2 columns
fig = make_subplots(rows=2, cols=1, subplot_titles=("MSLE", "Q2 Score"))

# # Assuming 'df' is your DataFrame containing the data
# states = df['State']
# mse = df['MSE']
# q2_score = df['q2_Score']

# Add the MSE bar plot to the first subplot
# fig.add_trace(go.Bar(x=state_names, y=mse, name='MSE'), row=1, col=1)

## Add the msle bar plot to the first subplot
fig.add_trace(go.Bar(x=state_names, y=msle, name='MSLE'), row=1, col=1)

# Add the q2 Score bar plot to the second subplot
fig.add_trace(go.Bar(x=state_names, y=q2, name='Q2 Score'), row=2, col=1)

## remove legend
fig.update_layout(showlegend=False)

# Update layout if necessary
fig.update_layout(height=600, width=1000, title_text="MSLE and Q2 Scores by State Using SVM with Subsampling ")

fig.write_image("//Users/ericliao/Desktop/WNV_project_files/WNV/CDC_data/human/result/SVM_each_state_subsampling/human_yearly_cdc_svm_each_state.png", scale=2)


# Create a subplot with 2 row and 2 columns
fig1 = make_subplots(rows=2, cols=1, subplot_titles=("Number of Data Rows", "Number of Non-0 WNNV Rows"))

## add the number of rows to the third subplot
fig1.add_trace(go.Bar(x=state_names, y=num_rows, name='Number of Data Rows'), row=1, col=1)

## add the number of non_0_num_rowrows to the fourth subplot
fig1.add_trace(go.Bar(x=state_names, y=non_0_num_rows, name='Number of Non-0 WNNV Rows'), row=2, col=1)

## remove legend
fig1.update_layout(showlegend=False)

# Update layout if necessary
fig1.update_layout(height=600, width=1000, title_text="Data Rows by State")

fig1.write_image("/Users/ericliao/Desktop/WNV_project_files/WNV/CDC_data/human/result/SVM_each_state_subsampling/basic_data_each_state.png", scale=2)


## plot only positive q2 score
state_mse_msle_q2_df = state_mse_msle_q2_df[state_mse_msle_q2_df["q2"] > 0]

fig2 = go.Figure(data=go.Bar(name='Q2 Score', x=state_mse_msle_q2_df["State"], y=state_mse_msle_q2_df["q2"]))

fig2.update_layout(title_text="Positive Q2 Score by State Using SVM with Subsampling")

fig2.write_image("/Users/ericliao/Desktop/WNV_project_files/WNV/CDC_data/human/result/SVM_each_state_subsampling/positive_q2_score_svm_each_state.png", scale=2)


# Create a subplot with 2 row and 2 columns
fig4 = make_subplots(rows=2, cols=1, subplot_titles=( "MSLE ", "Q2 Score"))

## add the number of non_0_num_rowrows to the fourth subplot
fig4.add_trace(go.Bar(x=state_mse_msle_q2_df["State"], y=state_mse_msle_q2_df["msle"]), row=1, col=1)

## add the Positive Q2 Score by State Using SVM
fig4.add_trace(go.Bar(x=state_mse_msle_q2_df["State"], y=state_mse_msle_q2_df["q2"]), row=2, col=1)

## remove legend
fig4.update_layout(showlegend=False)

# Update y-axis range for the first subplot
fig4.update_yaxes(title_text="MSLE", row=1, col=1, range=[0, 1])  # Example range, adjust as needed

# Update layout if necessary
fig4.update_layout(height=600, width=1000, title_text="Positive Q2 State Models Using SVM")

fig4.write_image("/Users/ericliao/Desktop/WNV_project_files/WNV/CDC_data/human/result/SVM_each_state_subsampling/positive_q2_msle_each_state_subsampling.png", scale=2)




