import numpy as np
import pandas as pd
from sklearn import linear_model

RANDOM_OFFSET = 0.5985

COLLEGE_1_RATING = 3.5
COLLEGE_2_RATING = 4.5

def generateDatasetArray(data_df):
    data_arr = np.zeros(len(data_df))
    for data in data_df.iterrows():
        index = data[0]
        row = data[1]
        data_arr[index] = RANDOM_OFFSET+row['knows']/row['heard']
    
    return data_arr
    
if __name__ == '__main__':
    """
        Evaluate colleges for a security rating based on the information in CSV
        Parameters in CSV
            - `methods`: Methods of Intrusion
            - `heard`: Has heard of the method
            - `knows`: Knows the method for sure
            - `college`: College Identification

        Security Awareness Rating
            - Linear Regression to generate a function to evaluate colleges
              on the given parameter and generate a rating.

              Use the ratio of knows and heard for each method as a single feature.

              sum{ method<knows>/method<heard> for method in all methods }

              dataset: Divide the methods on the basis of the college and
                       in the order described in the csv.
    """
    data_df = pd.read_csv('data.csv')
    college_1_df, college_2_df = data_df[data_df['college'] == 1].reset_index(drop=True), data_df[data_df['college'] == 2].reset_index(drop=True)

    print("Data of College 1")
    print(college_1_df)
    print("Data of College 2")
    print(college_2_df)
    print()

    college_1_dataset = generateDatasetArray(college_1_df)
    college_2_dataset = generateDatasetArray(college_2_df)
    
    print("Calculated Parameters for College 1")
    print(college_1_dataset)
    print("Calculated Parameters for College 2")
    print(college_2_dataset)
    print()

    X = [college_1_dataset, college_2_dataset]
    y = [COLLEGE_1_RATING, COLLEGE_2_RATING]

    reg = linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0])
    reg.fit(X, y)
    
    predictions = [round(val, 2) for val in reg.predict(X)]

    print("Rating of College 1: {}".format(predictions[0]))
    print("Rating of College 2: {}".format(predictions[1]))
    print()