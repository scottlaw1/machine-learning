from scipy.spatial import distance

def euc(a,b):
    return distance.euclidean(a,b)

class ScrappyKNN():
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = []
        for row in X_test:
            label = self.closest(row)
            predictions.append(label)
        return predictions

    def closest(self, row):
        best_dist = euc(row, self.X_train[0])
        best_index = 0
        for i in range(1, len(self.X_train)):
            dist = euc(row, self.X_train[i])
            if dist < best_dist:
                best_dist = dist
                best_index = i
        return self.y_train[best_index]

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