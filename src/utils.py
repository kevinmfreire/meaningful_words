'''
Script for splitting dataset into training/testing data.
'''

import base64

import pandas as pd
from sklearn.model_selection import train_test_split


def render_svg(svg):
    """Renders the given svg string."""
    b64 = base64.b64encode(svg.encode("utf-8")).decode("utf-8")
    html = r'<img src="data:image/svg+xml;base64,%s"/>' % b64
    return html


def train_test_data(label, feature, test_size=0.2, random_state=0):
    '''Split training data, and map labels to integer value.'''
    label_mapping = {"negative": 0, "neutral": 1, "positive": 2}

    # Split data into training and testing sets
    X = feature  # If uploading features as .npy files then X = feature.tolist()
    y = pd.Series(label).map(label_mapping)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print("Original set  ---> ", "feature size: ", X.shape, "label size", len(y))
    print(
        "Training set  ---> ",
        "feature size: ",
        X_train.shape,
        "label size",
        len(y_train),
    )
    print("Test set  --->  ",
          "feature size: ",
          X_test.shape, "label size",
          len(y_test)
          )

    return X_train, X_test, y_train, y_test
