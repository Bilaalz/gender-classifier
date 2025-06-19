from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# [height, weight, shoe size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37],
     [166, 65, 40], [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male',
     'male', 'female', 'female', 'female', 'male', 'male']

# Predict this new sample
sample = [[190, 70, 43]]

# Decision Tree Classifier
clf_tree = tree.DecisionTreeClassifier()
clf_tree.fit(X, Y)
pred_tree = clf_tree.predict(sample)
acc_tree = accuracy_score(Y, clf_tree.predict(X))

# K-Nearest Neighbors Classifier
clf_knn = KNeighborsClassifier()
clf_knn.fit(X, Y)
pred_knn = clf_knn.predict(sample)
acc_knn = accuracy_score(Y, clf_knn.predict(X))

# Support Vector Machine Classifier
clf_svm = SVC()
clf_svm.fit(X, Y)
pred_svm = clf_svm.predict(sample)
acc_svm = accuracy_score(Y, clf_svm.predict(X))

# Naive Bayes Classifier
clf_nb = GaussianNB()
clf_nb.fit(X, Y)
pred_nb = clf_nb.predict(sample)
acc_nb = accuracy_score(Y, clf_nb.predict(X))

# Print all predictions and accuracies
print("Decision Tree Prediction:", pred_tree[0], "| Accuracy:", acc_tree)
print("KNN Prediction:", pred_knn[0], "| Accuracy:", acc_knn)
print("SVM Prediction:", pred_svm[0], "| Accuracy:", acc_svm)
print("Naive Bayes Prediction:", pred_nb[0], "| Accuracy:", acc_nb)