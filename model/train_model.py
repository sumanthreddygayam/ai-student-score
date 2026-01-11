import pickle
import numpy as np
from sklearn.linear_model import LinearRegression

# Sample dataset
X = np.array([[1], [2], [3], [4], [5], [6], [7]])
y = np.array([35, 40, 50, 55, 65, 70, 80])

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
with open("model/model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved")
