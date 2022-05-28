from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier

from main import classification_summary, train_test_data, load_data

if __name__ == 'main':

    label_file = './data/label.npy'
    feature_file = './data/feature.npy'

    label, feature = load_data(label_file, feature_file)
    X_train, X_test, y_train, y_test = train_test_data(label, feature)

    # Building Logistic Regression Classifier
    log_reg_model = LogisticRegression()
    log_reg_model.fit(X_train, y_train)
    pred = log_reg_model.predict(X_test)
    pred_prob = log_reg_model.predict_proba(X_test)
    classification_summary(pred,pred_prob,'Logistic Regression (LR)')

    # Building  Decision Tree Classifier
    # DT_model = DecisionTreeClassifier()
    # DT_model.fit(X_train, y_train)
    # pred = DT_model.predict(X_test)
    # pred_prob = DT_model.predict_proba(X_test)
    # classification_summary(pred, pred_prob, 'Decision Tree Classifier (DT)')