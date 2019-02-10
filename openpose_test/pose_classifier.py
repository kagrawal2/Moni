import numpy as np
from sklearn import svm
from sklearn.model_selection import train_test_split
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
import pickle

sit = np.load('sitting.npy')
print(len(sit))

stand = np.load('standing.npy')
print(len(stand))

fall = np.load('fallen.npy')
print(len(fall))
# b = np.load('none.npy')
# print(len(b))

X = np.append(sit, stand, axis=0)
X = np.append(X, fall, axis=0)
Y = np.array([0]*len(sit) + [1]*len(stand) + [2]*len(fall))
print(X.shape)
print(Y.shape)

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=109) # 70% training and 30% test

#Create a svm Classifier
clf = svm.SVC(kernel='linear', probability=True) # Linear Kernel

#Train the model using the training sets
clf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)


# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:",metrics.precision_score(y_test, y_pred, average=None))
# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(y_test, y_pred, average=None))


filename = 'finalized_model.sav'
pickle.dump(clf, open(filename, 'wb'))

