import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from joblib import dump, load
import pickle

column_names = ["pregnancies", "glucose", "BP", "skinfold", "insulin", "bmi", "pedigree", "age", "class"]

data = pd.read_csv('data.csv', names=column_names)

# separates features and classes from the original data
X = data.iloc[:, :8]
Y = data["class"]

# splits data into testing and training samples
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

# checks how to normalize features
scalar = StandardScaler()
scalar.fit(x_train)

dump(scalar, 'std_scaler.bin', compress=True)

# normalizes features based on previous determination
x_train = scalar.transform(x_train)

# apply SVC on the dataset using kernel as rbf because it had the highest accuracy among other kernels
classification = svm.SVC(kernel='rbf')
classification.fit(x_train, y_train)
pickle.dump(classification, open('trainedModel.sav', 'wb'))

patient = np.array([[1., 200., 75., 40., 0., 45., 1.5, 20]])


classifier = pickle.load(open('trainedModel.sav', 'rb'))
scalar = load('std_scaler.bin')

patient = scalar.transform(patient)
pred = classifier.predict(patient)
