import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load the data from the 'communities.data' file into lists
data = []
with open('communities.data', 'r') as file:
    for line in file:
        # Split each line into a list of values
        line = line.strip().split(',')
        data.append(line)

# Extract and format the data
data = data[1:]  # Skip the header row
# Extract features (X) and target variable (y)
X = [[float(row[41]), float(row[45])] for row in data]  # PctUnemployed and PctBachDeg
y = [float(row[127]) for row in data]  # ViolentCrimesPerPop

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a linear regression model and fit it to the training data
model = LinearRegression()
model.fit(X_train, y_train)

# Get the coefficients and intercept of the model
coefficients = model.coef_
intercept = model.intercept_

# Print the coefficients and intercept
print("Intercept:", intercept)
print("Coefficient for PctUnemployed:", coefficients[0])
print("Coefficient for PctBachDeg:", coefficients[1])

# Interpret the coefficients
print("The intercept represents the expected ratio of violent crimes per population when both PctUnemployed and PctBachDeg are zero.")
print("The coefficient for PctUnemployed represents the change in the ratio of violent crimes per population for a one-unit change in the percentage of unemployed people, holding other factors constant.")
print("The coefficient for PctBachDeg represents the change in the ratio of violent crimes per population for a one-unit change in the percentage of people with a bachelor's degree or higher education, holding other factors constant.")

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model using mean squared error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
