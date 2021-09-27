import pandas as pd
import sklearn
from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv("../Data/Car Evaluation Data Set/car.data")

# Takes labels and encodes them into appropriate integer values i.e. Encoding
labelEncoder = preprocessing.LabelEncoder()

buying = labelEncoder.fit_transform(list(data["buying"]))
maintenance = labelEncoder.fit_transform(list(data["maint"]))
door = labelEncoder.fit_transform(list(data["door"]))
persons = labelEncoder.fit_transform(list(data["persons"]))
lugBoot = labelEncoder.fit_transform(list(data["lug_boot"]))
safety = labelEncoder.fit_transform(list(data["safety"]))

cls = labelEncoder.fit_transform(list(data["class"]))

predict = "class"

# Initialize x & y axes
X = list(zip(buying, maintenance, door, persons, lugBoot, safety))
y = list(cls)

# Categorizes data into training and testing data
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

# Initialises KNN Algorithm where K=10
model = KNeighborsClassifier(n_neighbors=10)

# Train model and test accuracy
model.fit(x_train, y_train)
acc = model.score(x_test, y_test)
print(f"Accuracy: {acc}")

predicted = model.predict(x_test)
names = ["unacc", "acc", "good", "vgood"]

# Print data in easy to read form
for x in range(len(x_test)):
    print(f" \tPredicted: {names[predicted[x]]} \tData: {x_test[x]} \tActual: {names[y_test[x]]}")
