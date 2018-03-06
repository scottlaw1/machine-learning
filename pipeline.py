#import a dataset
from sklearn import datasets
iris = datasets.load_iris()

X = iris.data #features
y = iris.target #labels

#think of a classifier as a function, where f(X) = Y

#split the datsets into training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = .5)

#choose a classifier
#from sklearn import tree
#my_classifier = tree.DecisionTreeClassifier()
#choose a different classifier
from sklearn.neighbors import KNeighborsClassifier
my_classifier = KNeighborsClassifier()

#train the classifier
my_classifier.fit(X_train, y_train)

#use the trained classifier to make predictions on test data
predictions = my_classifier.predict(X_test)
print(predictions)

#compute the accuracy of the predictions
from sklearn.metrics import accuracy_score
print(accuracy_score(y_test,predictions))