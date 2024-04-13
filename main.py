import datetime
import pandas as pd
import numpy as np
from joblib import dump, load
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score


if __name__ == '__main__':
    date_time = datetime.datetime.now()
    def score_details(scores):
        return f"Mean: {np.mean(scores)}\nStandard deviation: {np.std(scores)}"

    def add_result(details):
        choice  = str(input("Do you wnat to save & dump the results? (Y or N) "))
        if choice == "Y":
            with open('Results.txt') as f:
                data = f.read()
                if data == '':
                    with open('Results.txt', 'a') as f2:
                        f2.write("Models Result -\n\n")
                        f2.write(f"Day: {date_time.strftime("%x")} Date: {date_time.strftime("%X")}\n\n")
                        f2.write(f"{type(model).__name__} using cross_val_score with 10 parts\n")
                        f2.write(details +  "\n")
                        f2.write("_______________________________________________________________________________________________\n\n")
                else:
                    with open('Results.txt', 'a') as f3:
                        f3.write(f"Day: {date_time.strftime("%x")} Date: {date_time.strftime("%X")}\n\n")
                        f3.write(f"{type(model).__name__} using cross_val_score with 10 parts\n")
                        f3.write(details + "\n")
                        f3.write("_______________________________________________________________________________________________\n\n")
            dump(model, 'UrbanRentSolutions.joblib')
            print("Saved sucussfully in 'Results.txt'")
        else:
            print("Action denied")
        
                    
    def fit_transfor_dataFrame(df):
        my_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('my_standard', StandardScaler()),
        ])
        return my_pipeline.fit_transform(df)
    
    housing = pd.read_csv('Data.csv')
    train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
    
    # To find number of ROWS and COLUMNS in dataset.
    # print(f"Rows in training set: {len(train_set)}\nRows in testing set: {len(test_set)}")

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing['CHAS']):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]

    housing = strat_train_set.drop("MEDV", axis=1)
    housing_labels = strat_train_set['MEDV']
    housing_transformed = fit_transfor_dataFrame(housing)

    # Trying different algorithms to get least error
    
    # model = LinearRegression()
    # model = DecisionTreeRegressor()
    model = RandomForestRegressor()
    
    model.fit(housing_transformed, housing_labels)
    
    housing_predictions = model.predict(housing_transformed)
    mse = mean_squared_error(housing_labels, housing_predictions)
    rmse = np.sqrt(mse)
    scores = cross_val_score(model, housing_transformed, housing_labels, scoring='neg_mean_squared_error', cv=10)
    rmse_score = np.sqrt(-scores)
    
    detail = score_details(rmse_score)
    add_result(detail)
    