from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import joblib

from preprocessing import load_and_split

x_train, x_test, y_train, y_test = load_and_split()
model = LinearRegression()

model.fit(x_train, y_train)

predictions = model.predict(x_test)

mac = mean_squared_error(y_test,predictions)

print(f"Model MSE: {mac}")

joblib.dump(model,"boston_model.pkl")
print("Model saved Successfully!")

