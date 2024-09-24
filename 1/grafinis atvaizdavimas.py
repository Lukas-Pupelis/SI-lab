import numpy as np
import matplotlib.pyplot as plt

# 1. Define the bias term (you can change this value as needed)
b = -0.6  # Example value for b

# 2. Create a grid of x1 and x2 values
x1 = np.linspace(-10, 10, 400)
x2 = np.linspace(-10, 10, 400)
X1, X2 = np.meshgrid(x1, x2)

# 3. Define each inequality based on the system
# Inequality 1: -0.2x1 + 0.5x2 + b < 0
ineq1 = (-0.2 * X1 + 0.5 * X2 + b) < 0

# Inequality 2: 0.2x1 - 0.7x2 + b < 0
ineq2 = (0.2 * X1 - 0.7 * X2 + b) < 0

# Inequality 3: 0.8x1 - 0.8x2 + b >= 0
ineq3 = (0.8 * X1 - 0.8 * X2 + b) >= 0

# Inequality 4: 0.8x1 + x2 + b >= 0
ineq4 = (0.8 * X1 + X2 + b) >= 0

# 4. Combine all inequalities to find the feasible region
feasible_region = ineq1 & ineq2 & ineq3 & ineq4

# 5. Plotting
plt.figure(figsize=(10, 10))

# a. Shade the feasible region where all inequalities are true
plt.contourf(X1, X2, feasible_region, levels=[0.5, 1], colors=['#A0FFA0'], alpha=0.5, label='Feasible Region')

# b. Plot boundary lines for each inequality
# Function to compute y based on the equality (boundary line) for a given inequality
def plot_boundary(coeff_x1, coeff_x2, b, label, color):
    if coeff_x2 != 0:
        y = (-coeff_x1 * x1 - b) / coeff_x2
        plt.plot(x1, y, label=label, color=color)
    else:
        # Vertical line when coeff_x2 is zero
        x = -b / coeff_x1
        plt.axvline(x=x, label=label, color=color)

# Plot each boundary line
plot_boundary(-0.2, 0.5, b, '-0.2x₁ + 0.5x₂ + b = 0', 'red')      # Inequality 1
plot_boundary(0.2, -0.7, b, '0.2x₁ - 0.7x₂ + b = 0', 'blue')      # Inequality 2
plot_boundary(0.8, -0.8, b, '0.8x₁ - 0.8x₂ + b = 0', 'green')     # Inequality 3
plot_boundary(0.8, 1, b, '0.8x₁ + x₂ + b = 0', 'purple')          # Inequality 4

# c. Add labels, title, and legend
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('Visualization of the Inequality System')
plt.legend()

# d. Set plot limits
plt.xlim([-10, 10])
plt.ylim([-10, 10])

# e. Add grid for better readability
plt.grid(True, linestyle='--', alpha=0.7)

# 6. Show the plot
plt.show()