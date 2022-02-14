from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

"""
A file of pre-defined functions for use in the project.
Operations that can be generalised across different datasets and are reusable
are defined in this file
"""
# check if key exists in a list of dictionaries
def key_exists(k1, k2, list_of_dicts):
    for d in list_of_dicts:
        if k1 == d['Column 1'] and k2 == d['Column 2'] or k1 == d['Column 2'] and k2 == d['Column 1']:
            return True
    return False

# find columns with correlation greater than threshold value
def get_good_correlation(df, threshold=0.15):
    cor = df.corr()
    good_cor = []
    for i in cor.columns:
        for j in cor.columns:
            if i != j and (cor[i][j] > threshold or cor[i][j] < -threshold) and key_exists(i, j, good_cor) == False:
                good_cor.append(
                    {
                        'Column 1': i,
                        'Column 2': j,
                        'cor': cor[i][j]
                    }
                )
    return cor, good_cor

#normalise numerical values and label encode non-numerical values
def normalise_and_encode(data, data_y, **kwargs):
    for col in data.columns:
        if data[col].dtype == 'object':
            # if we are using a decision tree classifier, then label encode only the target column
            if 'dct' in kwargs:
                # if column to predict is non numerical, then use label encoding
                if col == data_y:
                    le = preprocessing.LabelEncoder()
                    data[col] = le.fit_transform(data[col])
            # label encode all non-numerical columns
            if kwargs['encoding'] == 'Label Encoding':
                le = preprocessing.LabelEncoder()
                data[col] = le.fit_transform(data[col])
        # normalise numerical data         
        elif data[col].dtype != 'object' and col != data_y:
            min_max_scaler = preprocessing.StandardScaler()
            x_scaled = min_max_scaler.fit_transform(data[col].values.reshape(-1, 1))
            data[col] = x_scaled

    # one hot encode categorical data for decision tree classifier
    if kwargs['encoding'] == 'One Hot Encoding':
        data = pd.get_dummies(data)
    return data

"""
Test - Train Split boilerplate
"""
def split_data(x,y):
    x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.3)
    return x_train, x_test, y_train, y_test

"""
Decison Tree Classifier Function
    simple decisions tree classifer
    @returns: model, mean accuracy score, model, report
"""
def decision_tree_classifier(df, data_y):
    x = df.drop(data_y, axis=1)
    y = df[data_y]
    x_train, x_test, y_train, y_test = split_data(x,y)
    model_dt = DecisionTreeClassifier(criterion='gini',random_state=100,max_depth=6, min_samples_leaf=8)
    model_dt.fit(x_train, y_train)
    y_pred = model_dt.predict(x_test)
    report = classification_report(y_test, y_pred, labels = [0,1], output_dict=True)
    return model_dt, y_pred, model_dt.score(x_test,y_test), report

"""
Random Forest Classifier Function
    simple classifer
    @returns: model, mean accuracy score, model, report
"""
def random_forest_classifier(df, data_y):
    x = df.drop(data_y, axis=1)
    y = df[data_y]
    x_train, x_test, y_train, y_test = split_data(x,y)
    model_rf = RandomForestClassifier(criterion='entropy',random_state=100,max_depth=20, min_samples_leaf=5, n_estimators=40)
    model_rf.fit(x_train, y_train)
    y_pred = model_rf.predict(x_test)
    report = classification_report(y_test, y_pred, labels = [0,1], output_dict=True)
    return model_rf, y_pred , model_rf.score(x_test,y_test) , report