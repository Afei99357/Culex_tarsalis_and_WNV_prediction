import pandas as pd
import numpy as np


class Corpus:
    def __init__(self):
        # Load dataset
        # data = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/DATA/"
        #                    "human_neuroinvasive_with_extreme_weather_with_county_seat_modify.csv", index_col=False)
        #
        # # drop rows contains nan
        # data = data.dropna()
        #
        # # find Counties in southern california
        # southern_california_counties = ["Los Angeles", "San Diego", "Orange", "Riverside", "San Bernardino", "Kern", "Ventura",
        #                                 "Santa Barbara", "San Luis Obispo", "Imperial"]
        #
        # data = data[data["County"].isin(southern_california_counties) | data["State"].isin(['North Dakota', 'South Dakota', 'Colorado'])]
        #
        # # only on horse
        # # data = data[data["SET"] == "VET"]
        # # only on human
        # data = data[data["SET"] == "HUMAN"]

        ## pca data
        data = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/ebirds_results/Data/"
                           "PCA_3pc_wnv_all_counties_ebirds.csv", index_col=False)

        # split into train and test
        year = data.pop("Year")

        # # save test dataset
        # df_test_save = data[(year >= 2019) & (year < 2022)].copy()
        # df_test_save.to_csv("/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/2023_02_13_to_2023_02_20/results/test_dataset_pca.csv", index=True)

        # data = data.drop(["FIPS", "County", "State", "State_Code", "Year",
        #                   # "Non_Neural_WNV_COUNT", "Neural_WNV_Count",
        #                   "County_Centroid_Latitude", "County_Centroid_Longitude", "County_Seat_Latitude",
        #                   "County_Seat_Longitude", "County_Seat", "Processed_Flag_Land_Use", 'SET',
        #                   # 'Poverty_Estimate_All_Ages'
        #                   ], axis=1)

        train = data[(year < 2019) & (year >= 2003)]
        test = data[(year >= 2019) & (year < 2022)]

        # save test dataset
        # test.to_csv("/Users/ericliao/Desktop/WNV_project_files/Data_for_Machine_Learning/2023_02_13_to_2023_02_20/results/test_dataset.csv")

        # Get labels
        self.train_labels = train.pop("WNV_Rate_Neural_Without_99_21").values
        self.test_labels = test.pop("WNV_Rate_Neural_Without_99_21").values

        # get the column names
        self.train_column_names = train.columns
        self.test_column_names = test.columns

        # get the shape of train and test for neural network input layer
        self.train_column_number = train.shape[1]
        self.test_column_number = test.shape[1]

        self.land_change_train = train.pop("Land_Change_Count_Since_1992").to_numpy()
        self.land_change_test = test.pop("Land_Change_Count_Since_1992").to_numpy()

        # get categorical feature 'land use class' and "land change"
        self.land_use_class_train = train.pop("Land_Use_Class").to_numpy()
        self.land_use_class_test = test.pop("Land_Use_Class").to_numpy()

        # given epslon to avoid divide by zero
        epslon = 1e-6

        # normalize each feature in the train and test
        for column in train.columns:
            train[column] = (train[column] - train[column].mean()) / train[column].std() + epslon

        for column in test.columns:
            test[column] = (test[column] - test[column].mean()) / test[column].std() + epslon

        # # store the standard deviation and mean for train_labels and test_labels
        self.train_labels_std = self.train_labels.std()
        self.train_labels_mean = self.train_labels.mean()
        self.test_labels_std = self.test_labels.std()
        self.test_labels_mean = self.test_labels.mean()

        # normalize the train_labels and test_labels
        self.train_labels = (self.train_labels - self.train_labels_mean) / self.train_labels_std + epslon
        self.test_labels = (self.test_labels - self.test_labels_mean) / self.test_labels_std + epslon

        self.train = train.to_numpy()
        self.test = test.to_numpy()

        # prepare for neural network
        self.input_non_categorical_shape = (train.shape[1],)

        # Calculate the number of unique values in the categorical features for embedding input layer
        self.unique_categorical_land_use = np.unique(np.concatenate((self.land_use_class_train, self.land_use_class_test), axis=0))
        # use search sorted to convert input_categorical_land_use_shape to continuous number
        self.input_land_use_train = np.searchsorted(self.unique_categorical_land_use, self.land_use_class_train)
        self.input_land_use_test = np.searchsorted(self.unique_categorical_land_use, self.land_use_class_test)

        self.unique_categorical_land_change_shape = np.unique(np.concatenate((self.land_change_train, self.land_change_test), axis=0))

        ## add sample weight by add 1 to each sample disease count then do square root
        self.sample_weights = np.sqrt(self.train_labels + 1)

        # weights = np.power(np.abs(self.train_labels + 1), 2)
        # self.sample_weights = self.train_labels + 1 / np.max(weights)

