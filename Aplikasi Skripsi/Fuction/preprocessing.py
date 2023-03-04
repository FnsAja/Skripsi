from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split

cancer = datasets.load_breast_cancer()

# 70% train, 30% test -> (test size)
X_train, X_test, Y_train, Y_test = train_test_split(cancer.data, cancer.target, test_size=0.3, random_state=110)

# SVM Model
clf = svm.SVC(kernel='rbf')

# Train Model using fit()
clf.fit(X_train, Y_train)

# Predict
y_pred = clf.predict(X_test)

print("Accuracy : ", metrics.accuracy_score(Y_test, y_pred))
print("Precision : ", metrics.precision_score(Y_test, y_pred))
print("Recall : ", metrics.recall_score(Y_test, y_pred))