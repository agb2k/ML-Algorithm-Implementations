import sklearn
from sklearn import datasets
from sklearn import svm
from sklearn import metrics

cancer = datasets.load_breast_cancer()

x = cancer.data
y = cancer.target

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

classes = ["Malignant", "Benign"]

classifier = svm.SVC(kernel="linear", C=1)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

accuracy = metrics.accuracy_score(y_test, y_pred)
print(accuracy)

# Better than KNN for more features
