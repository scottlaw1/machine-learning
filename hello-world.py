from sklearn import tree
#original features and labels
#features = [[140, "smooth"], [130, "smooth"], [150, "bumpy"], [170, "bumpy"]]
#labels = ["apple","apple","orange","orange"]

#switching text features and labels to numeric values because scikit-learn requires this
features = [[140, 1], [130, 1], [150, 0], [170, 0]]
labels = [0,0,1,1]

#clf is shorter for classifier
clf = tree.DecisionTreeClassifier() #Classifier is untrained here
clf = clf.fit(features, labels) #Now the classifier is trained

print("0 is apple")
print("\n1 is orange")
print("\nTrained algorithm given a 160g bumpy fruit predicts:")
print(clf.predict([[160,0]]))