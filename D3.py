#################################
### HW4, P4 CSDS435           ###
### Maryam Ghasemian (mxg708) ###
#################################

from sklearn import tree, metrics
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier



def importdata():
    tennis_data = pd.read_csv (
        'G:/My Drive/Spring 2021/CSDS435/assignments/HW4/p4/tennis.csv',
        sep=',')

    # Printing the dataswet shape
    print ("Dataset Length: ", len (tennis_data))
    print ("Dataset Shape: ", tennis_data.shape)

    # Printing the dataset obseravtions
    print ("Dataset: ", tennis_data.head ())
    return tennis_data


# Function to split the dataset
def splitdataset(tennis_data):
    feature_cols = ['outlook', 'temperature', 'humidity', 'windy']
    X = tennis_data[feature_cols]  # Features
    Y = tennis_data.play  # Target variable

    # Splitting the dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split (
        X, Y, test_size=0.3, random_state=10)

    return X, Y, X_train, X_test, y_train, y_test


# Function to perform training with giniIndex.
def train_using_gini(X_train, X_test, y_train):
    # Creating the classifier object
    clf_gini = DecisionTreeClassifier (criterion="gini",
                                       random_state=1, max_depth=3, min_samples_leaf=5)

    # Performing training
    clf_gini.fit (X_train, y_train)
    return clf_gini


# Function to perform training with entropy.
def tarin_using_entropy(X_train, X_test, y_train):
    # Decision tree with entropy
    clf_entropy = DecisionTreeClassifier (
        criterion="entropy", random_state=1,
        max_depth=3, min_samples_leaf=5)

    # Performing training
    clf_entropy.fit (X_train, y_train)
    return clf_entropy




def prediction(X_test, clf_object):
    # Predicton on test with giniIndex
    y_pred = clf_object.predict (X_test)
    print ("Predicted values:")
    print (y_pred)
    return y_pred


def main():

    feature_cols = ['outlook', 'temperature', 'humidity', 'windy']
    data = importdata ()
    X, Y, X_train, X_test, y_train, y_test = splitdataset (data)
    clf_gini = train_using_gini (X_train, X_test, y_train)
    clf_entropy = tarin_using_entropy (X_train, X_test, y_train)


    print ("\n Results Using Gini Index:")

    # Prediction using gini
    y_pred_gini = prediction (X_test, clf_gini)
    print("Accuracy: ",metrics.accuracy_score (y_test, y_pred_gini))

    print ("\n Results Using Information Gain:")
    y_pred_entropy = prediction (X_test, clf_entropy)
    print("Accuracy: ",metrics.accuracy_score(y_test, y_pred_entropy))



# Calling main function
if __name__ == "__main__":
    main ()


