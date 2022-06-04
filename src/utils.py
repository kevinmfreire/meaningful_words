import pandas as pd
from sklearn.model_selection import train_test_split

def train_test_data(label, feature, test_size=0.2, random_state=0):
    #Assign labels to target value
    label_mapping = {'negative':0, 'neutral':1, 'positive':2}

    # Split data into training and testing sets
    X = feature                                             # If uploading features as .npy files then X = feature.tolist()
    y = pd.Series(label).map(label_mapping)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    print('Original set  ---> ','feature size: ',X.shape,'label size',len(y))
    print('Training set  ---> ','feature size: ',X_train.shape,'label size',len(y_train))
    print('Test set  --->  ','feature size: ',X_test.shape,'label size',len(y_test))

    return X_train, X_test, y_train, y_test