from sklearn import ensemble, metrics
import pandas as pd
from dataclasses import dataclass
import numpy as np
import shap

@dataclass
class Split:
    train: pd.DataFrame
    train_labels: np.ndarray
    test: pd.DataFrame
    test_labels: np.ndarray

    def grade(self, predict_y):
        mse = metrics.mean_squared_error(self.test_labels, predict_y)
        r2 = metrics.r2_score(self.test_labels, predict_y)

        # also show a histogram of errors
        errors = predict_y - self.test_labels
        hist, bin_edges = np.histogram(errors, bins=10)
        print(f"histogram of errors:")
        for i in range(len(hist)):
            print(f"{bin_edges[i]:02.02f}-{bin_edges[i+1]:02.02f}: {hist[i]}")

        return mse, r2, errors

class Dataset:
    def __init__(self):
        # Load the dataset into a Pandas DataFrame
        data = pd.read_csv("/Users/ericliao/Desktop/WNV_project_files/WNV/california/CA_13_county_dataset/CA_13_counties_04_23_no_impute.csv",
                           index_col=False,
                           header=0)

        # Get the Date column to a new dataframe
        date = data.pop("Date")

        # Drop columns that are not features and drop target
        data = data.drop([
            "Year",
            "County",
            "Latitude",
            "Longitude",
            "Total_Bird_WNV_Count",
            "Mos_WNV_Count",
            "Horse_WNV_Count",
        ], axis=1)

        # Drop columns if all the values in the columns are the same or all nan
        data = data.dropna(axis=1, how='all')

        # Reindex the data
        data = data.reset_index(drop=True)

        # Print 0 variance columns
        print(data.columns[data.var() == 0])

        # Check if any columns have zero variance and drop the columns
        data = data.loc[:, data.var() != 0]

        # Add the Date column back to the data
        data["Date"] = date

        # Convert "Date" column to datetime
        data["Date"] = pd.to_datetime(data["Date"])

        # Get the unique years and sort them
        self.years = data["Date"].dt.year.unique()
        self.years.sort()

        ## impute any missing in Human_Disease_Count with 0
        data["Human_Disease_Count"] = data["Human_Disease_Count"].fillna(0)
        self.data = data

    def split(self, year: int) -> Split:
        train = self.data[self.data['Date'].dt.year < year].copy()
        test = self.data[(self.data['Date'].dt.year == year)].copy()

        # Drop rows if they have nan values for both train and test data
        train = train.dropna().reset_index(drop=True)
        test = test.dropna().reset_index(drop=True)

        # Get labels
        train_labels = train.pop("Human_Disease_Count").values
        test_labels = test.pop("Human_Disease_Count").values

        # Remove unnecessary columns
        train.drop(["Month", "FIPS", "Date"], axis=1, inplace=True)
        test.drop(["Month", "FIPS", "Date"], axis=1, inplace=True)

        return Split(train, train_labels, test, test_labels)


def train_year(split: Split):
    # HGBR
    hgbr = ensemble.HistGradientBoostingRegressor(max_depth=20)

    hgbr.fit(split.train, split.train_labels)

    y_predict = hgbr.predict(split.test)
    mse, r2, errors = split.grade(y_predict)

    # For the worst prediction, show the shap values
    worst = np.argmax(np.abs(errors))
    explanation = shap.TreeExplainer(hgbr)(split.test)
    plot = shap.plots.bar(explanation)
    raise ValueError("stop here")
    return mse, r2


def train_all(dataset: Dataset):
    results = []
    for year in dataset.years[1:]:
        # Print the prediction year with r2
        mse, r2 = train_year(dataset.split(year))
        print("predict year: ", year, ", Q2: ", r2)
        results.append((year, mse, r2))
    years, mses, r2s = list(zip(*results))
    results = pd.DataFrame({
        "Year": years,
        "MSE": mses,
        "R2": r2s
    })

    print(results)


if __name__ == "__main__":
    #train_all(Dataset())
    train_year(Dataset().split(2020))

