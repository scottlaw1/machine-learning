import random

class ScrappyKNN():
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []
        for row in X_test:
            label = random.choice(self.y_train)
            predictions.append(label)
        return predictions

#import a dataset
from sklearn import datasets
iris = datasets.load_iris()

X = iris.data #features
y = iris.target #labels

#think of a classifier as a function, where f(X) = Y

#split the datsets into training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = .5)

#use custom classifier
my_classifier = ScrappyKNN()

#train the classifier
my_classifier.fit(X_train, y_train)

#use the trained classifier to make predictions on test data
predictions = my_classifier.predict(X_test)
print(predictions)

#compute the accuracy of the predictions
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,predictions))