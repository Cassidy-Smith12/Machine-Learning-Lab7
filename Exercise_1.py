import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

data = pd.read_csv('wdbc.data.csv', header=None)
data = data.iloc[:, 1:]

y = data.iloc[:, 0]
X = data.iloc[:, 1:]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("Variance ratio:", pca.explained_variance_ratio_)

plt.figure(figsize=(10, 7))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y.map({'M': 1, 'B': 0}), cmap='viridis', alpha=0.5)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA of Breast Cancer Dataset')

data_point = np.array([7.76, 24.54, 47.92, 181, 0.05263, 0.04362, 0, 0, 0.1587, 0.05884, 0.3857, 1.428, 2.548, 19.15, 0.007189, 0.00466, 0, 0, 0.02676, 0.002783, 9.456, 30.37, 59.16, 268.6, 0.08996, 0.06444, 0, 0, 0.2871, 0.07039])
data_point_scaled = scaler.transform(data_point.reshape(1, -1))
data_point_pca = pca.transform(data_point_scaled)
plt.scatter(data_point_pca[0, 0], data_point_pca[0, 1], color='red', marker='x', s=100, label='Data Point')

log_reg = LogisticRegression()
log_reg.fit(X_pca, y.map({'M': 1, 'B': 0}))

x_values = np.linspace(X_pca[:, 0].min(), X_pca[:, 0].max(), 100)
y_values = -(log_reg.intercept_ + log_reg.coef_[0][0] * x_values) / log_reg.coef_[0][1]
plt.plot(x_values, y_values, color='blue', label='Decision Boundary')

plt.legend()
plt.show()

prediction = log_reg.predict(data_point_pca)
print("Predicted class for the data point:", 'Malignant' if prediction[0] == 1 else 'Benign')
