import json
import pickle
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Training data
np.random.seed(42)
X = np.random.rand(100, 3)
y = np.random.rand(100)

# Train model
model = LinearRegression()
model.fit(X, y)

# Evaluate
y_pred = model.predict(X)
mse = mean_squared_error(y, y_pred)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save metrics
with open("metrics.json", "w") as f:
    json.dump({"mse": mse}, f)

print(f"Training complete. MSE: {mse:.4f}")
