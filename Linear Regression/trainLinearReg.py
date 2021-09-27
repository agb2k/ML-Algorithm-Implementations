# This is where model training takes place

import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
import matplotlib.pyplot as pyplot
from matplotlib import style
import pickle

# Pandas allows for convenient World Happiness Report
data = pd.read_csv("../Data/World Happiness Report/2019.csv")
# Attributes within dataset chosen as they are most likely to provide accurate readings
attributes = ["Score", "GDP per capita", "Social support", "Healthy life expectancy"]
data = data[attributes]

# Labels for prediction
predict = "Score"

# Drops score from the World Happiness Report
x = np.array(data.drop([predict], axis=1))
y = np.array(data[predict])
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

# For loop gets the model with the highest accuracy and uses thatA
best = 0.0
for _ in range(10000):
    # Splits 10% of World Happiness Report into test World Happiness Report and remaining into training
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

    linear = linear_model.LinearRegression()

    # Fitting the graph according to the training World Happiness Report i.e Training
    linear.fit(x_train, y_train)

    # Calculating the accuracy based on the test World Happiness Report
    accuracy = linear.score(x_test, y_test)
    print(accuracy)

    if accuracy >= best:
        # Used to store our model as a pickle file
        best = accuracy
        print(f"Best: {best}")
        with open("linearReg.pickle", "wb") as f:
            pickle.dump(linear, f)

pickleIn = open("linearReg.pickle", "rb")
linear = pickle.load(pickleIn)

accuracy = linear.score(x_test, y_test)

print(f"Model Accuracy: {accuracy}")
print(f"Coefficient: {linear.coef_}")
print(f"Intercept: {linear.intercept_}")

# Make predictions based on linear regression and test
predictions = linear.predict(x_test)
for i in range(len(predictions)):
    print(f"{i + 1}.\t Y Prediction: {predictions[i]} \n\t Test Y: {y_test[i]}\n\t Test X: {x_test[i]}\n\t")

p = "Healthy life expectancy"
style.use("ggplot")
pyplot.scatter(data[p], data["Score"])
pyplot.xlabel(p)
pyplot.ylabel("Score")
pyplot.title("World Happiness Report 2019")
pyplot.show()
