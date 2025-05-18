import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB

data = pd.read_csv('golf.csv')

print(data.head())

label_encoders = {}
for column in data.columns:
    label_encoders[column] = LabelEncoder()
    data[column] = label_encoders[column].fit_transform(data[column])

X = data.drop('PlayGolf', axis=1)
y = data['PlayGolf']

nb_classifier = GaussianNB()
nb_classifier.fit(X, y)

data_points = [
    ['Rainy', 'Hot', 'High', 'True'],
    ['Sunny', 'Mild', 'Normal', 'False'],
    ['Sunny', 'Cool', 'High', 'False']]

data_points_encoded = []
for point in data_points:
    encoded_point = [label_encoders[column].transform([value])[0] for column, value in zip(X.columns, point)]
    data_points_encoded.append(encoded_point)

predictions = nb_classifier.predict(data_points_encoded)

predictions_labels = label_encoders['PlayGolf'].inverse_transform(predictions)

for point, prediction in zip(data_points, predictions_labels):
    print(f"Data point: {point} -> Prediction: {prediction}")
