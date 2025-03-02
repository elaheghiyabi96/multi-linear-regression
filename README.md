# multi-linear-regression
A basic implementation of multi linear regression using gradient descent.  Synthetic housing price data is generated based on area and number of rooms.  Includes data visualization and model training from scratch using NumPy and Matplotlib
#MachineLearning #LinearRegression #GradientDescent #Python #NumPy #Matplotlib  
#DataScience #ArtificialData #Regression #DeepLearning #AI #HousingPrices  
#DataVisualization #ML #SyntheticData #Coding #OpenSource  
import numpy as np
import matplotlib.pyplot as plt

# Number of data points
n = 200  

# Generate house area (X1) from a normal distribution with mean=175 and std=30
X1 = np.random.normal(loc=175, scale=30, size=n)  

# Clip values to keep them within the range [50, 300]
X1 = np.clip(X1, 50, 300)  

# Generate the number of rooms (X2) as random integers between 1 and 6
X2 = np.random.randint(1, 7, size=n)  

# Generate house prices using a linear equation with some noise
price = X1 * 200 + X2 * 100 + 500 + np.random.normal(0, 3000, size=n)  

# Initialize parameters (weights and bias) randomly with small values
w1 = np.random.randn() * 0.1   # Weight for X1 (house area)
w2 = np.random.randn() * 0.1   # Weight for X2 (number of rooms)
b = np.random.randn() * 0.1    # Bias term

# Set learning rate and number of iterations for gradient descent
learning_rate = 0.00001
iterations = 1000

# Perform gradient descent to optimize weights and bias
for i in range(iterations):
    # Compute predictions using the current weights and bias
    predictions = w1 * X1 + w2 * X2 + b
    
    # Compute gradients (partial derivatives of the loss function)
    dw1 = (2/n) * np.sum((predictions - price) * X1)  # Gradient for w1
    dw2 = (2/n) * np.sum((predictions - price) * X2)  # Gradient for w2
    db = (2/n) * np.sum((predictions - price))        # Gradient for bias
    
    # Update parameters using gradient descent
    w1 -= learning_rate * dw1
    w2 -= learning_rate * dw2
    b -= learning_rate * db
    
    # Compute Mean Squared Error (MSE) loss
    Loss = (1/n) * np.sum((predictions - price) ** 2)
    
    # Print loss every 50 iterations to monitor training progress
    if i % 50 == 0:
        print(f"Iteration {i}: Loss {Loss}")

# Compute final predicted values using trained weights and bias
y = w1 * X1 + w2 * X2 + b

# Plot the data points (house area vs price)
plt.scatter(X1, price, alpha=0.5, label='Data')

# Plot the fitted regression line
plt.plot(X1, y, label='Fitted Line', color='r')

# Add legend to the plot
plt.legend()

# Show the plot
plt.show()
