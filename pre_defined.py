from sklearn import preprocessing

"""
A file of pre-defined functions for use in the project.
Operations that can be generalised across different datasets and are reusable
are defined in this file
"""

#normalise numerical values and label encode non-numerical values
def normalise_and_label_encode(data):
    for col in data.columns:
        if data[col].dtype == 'object':
            le = preprocessing.LabelEncoder()
            data[col] = le.fit_transform(data[col])
        elif data[col].dtype != 'object':
            min_max_scaler = preprocessing.MinMaxScaler()
            x_scaled = min_max_scaler.fit_transform(data[col].values.reshape(-1, 1))
            data[col] = x_scaled
    return data