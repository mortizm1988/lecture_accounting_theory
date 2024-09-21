
# %%
import numpy as np
from scipy.optimize import linprog
import pandas as pd

# Parameters
m = np.arange(0, 2.1, 0.1)
N = len(m)
theta = 0
x_max = 10

# Adjusted probabilities (for N=3)
g = np.array([1/N] * N)

# Objective function: maximize sum of (x_i - z_i) * g_i
# Notice this is equivalent to minimiza  sum of z_i * g_i -x_i * g_i 
# c is a one-dimensional vector of size 2N: [x_1, z_1, x_2, z_2, ..., x_N, z_N]
c = np.zeros(2 * N)
for i in range(N):
    c[2 * i] = -g[i]  # Coefficient for x_i
    c[2 * i + 1] = g[i]  # Coefficient for z_i

# Inequality constraints for participation and incentive compatibility
A = []
b = []

# %%
# Participation constraints: z_i - m_i * x_i >= theta (agent's minimal utility constraint)
# The constraint is a one-dimensional vector of size 2N to multiply with [x_1, z_1, x_2, z_2, ..., x_N, z_N]
for i in range(N):
    constraint = [0] * (2 * N)  # Create a 2N-dimensional row, for matching dimension with ICCs
    constraint[2 * i] = -m[i]   # Coefficient for x_i
    constraint[2 * i + 1] = 1   # Coefficient for z_i
    A.append(constraint)
    b.append(theta)

# Incentive compatibility constraints I: [z_i - m_i * x_i] -[z_{i-1} - m_{i} * x_{i-1}] >= 0 
# The constraint is a one-dimensional vector of size 2N to multiply with [x_1, z_1, x_2, z_2, ..., x_N, z_N]
for i in range(1, N):
    constraint = [0] * (2 * N)  # Create a 2N-dimensional row
    constraint[2 * i] = -m[i]  # Coefficient for x_i
    constraint[2 * i + 1] = 1  # Coefficient for z_i
    constraint[2 * (i - 1)] = m[i]  # Coefficient for x_{i-1}
    constraint[2 * (i - 1) + 1] = -1  # Coefficient for z_{i-1}
    A.append(constraint)
    b.append(0)

# Incentive compatibility constraints II: [z_i - m_i * x_i]-[z_{i+1} - m_{i} * x_{i+1}] >= 0 
for i in range(N - 1):
    constraint = [0] * (2 * N)  # Create a 2N-dimensional row
    constraint[2 * i] = -m[i]  # Coefficient for x_i
    constraint[2 * i + 1] = 1  # Coefficient for z_i
    constraint[2 * (i + 1)] = m[i]  # Coefficient for x_{i+1}
    constraint[2 * (i + 1) + 1] = -1  # Coefficient for z_{i+1}
    A.append(constraint)
    b.append(0)

# Convert A and b to numpy arrays for linprog function: A*constraints <= b
A = np.array(A) * -1  # Multiply A by -1 to convert the inequality to the form Ax <= b
b = np.array(b) * -1  # Multiply b by -1

# %%
# Bounds: 0 <= x_i <= x_max, 0 <= z_i
x_bounds = [(0, x_max)] * N
z_bounds = [(0, None)] * N
bounds = []
for i in range(N):
    bounds.append(x_bounds[i])
    bounds.append(z_bounds[i])

# Solve the linear programming problem using the revised formulation
res = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method='highs')

# %%
# Check if the optimization was successful
if res.success:
    x = res.x[::2]  # Extract the budget values from the solution, they are in even indices
    z= res.x[1::2]
    # the slack is transfers{i} -m{i}*cashflow{i}
    slack =  z- m *   x
    # Create a dataframe for the results
    df = pd.DataFrame({'m': m, 'cash_flow': x, 'transfers': z, 'slack': slack})
    # Print the results as markdown
    print(df.to_markdown())
else:
    print("Optimization failed. Check the constraints or problem formulation.")
# %%
