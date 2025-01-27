from FFCA_wrap import FFCA_wrap
from Grid import Grid, Pos
from metrics import order_parameter, mean_order_parameter, plot_order_parameter
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def lane_formation(rows, cols, Ntot, steps, ffca):
    # Calculate the mean phi for a random distribution of agents
    random_phi_values = []
    for _ in range(500):
        ffca = FFCA_wrap(rows, cols, Ntot)
        N1, N2 = ffca.agents_in_row(ffca.structure)
        random_phi_values.append(order_parameter(Ntot, N1, N2))
    phi_zero = np.mean(random_phi_values)

    ffca = FFCA_wrap(rows, cols, Ntot)
    phi_values = np.zeros(steps)

    for i in range(steps):
        current_phi = order_parameter(Ntot, *ffca.agents_in_row(ffca.structure))
        current_mean_phi = mean_order_parameter(current_phi, phi_zero)
        phi_values[i] = current_mean_phi
        ffca.step()
    return phi_values


steps = 1000
number_agents = 50
number_rows = 10
number_cols = 100
"""
ffca = FFCA_wrap(number_rows, number_cols, number_agents, spawn_rate=0.025,
                 conflict_resolution_rate=0, alpha=0.3, delta=0.3,
                 static_field_strength=2.5, dynamic_field_strength=3,
                 horizontal_bias=50)  # Adjust parameters as necessary

phi_values = lane_formation(number_rows, number_cols, number_agents, steps, ffca)


# Control values are the values defined in the article

# Different horizontal bias values 
horizontal_bias = [1, 10, 25, 50, 75, 100]
phi_lists = []
for bias in horizontal_bias:
    ffca = FFCA_wrap(number_rows, number_cols, number_agents, spawn_rate=0.025,
                 conflict_resolution_rate=0.5, alpha=0.3, delta=0.3,
                 static_field_strength=2.5, dynamic_field_strength=5,
                 horizontal_bias=bias)  
    phi_lists.append(lane_formation(number_rows, number_cols, number_agents, steps, ffca))

# Make a plot with all horizontal bias values (phi values vs iterations)

plt.figure(figsize=(10, 6))
for i, phi_values in enumerate(phi_lists):
    plt.plot(range(steps), phi_values, linestyle='-', label=f'Horizontal Bias: {horizontal_bias[i]}')
plt.xlabel("Iterations", fontsize=14)
plt.ylabel("Order Parameter", fontsize=14)
plt.title("Order Parameter vs Iterations", fontsize=16)
plt.grid(True)
plt.legend()
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig("order_parameter_vs_horizontal_bias.pdf")
plt.show()

# Save values in a CSV file with horizontal bias as column headers
data = np.array(phi_lists).T  # Transpose to align each bias as a column
columns = [f"Horizontal Bias {bias}" for bias in horizontal_bias]

# Create a DataFrame 
df = pd.DataFrame(data, columns=columns)

# Save the DataFrame to a CSV file
df.to_csv("order_parameter_vs_horizontal_bias.csv", index=False)
 """

# Different density values
total_num_cells = number_rows * number_cols

density_values = [0.05, 0.1, 0.15, 0.2, 0.25]
number_agents_list = [int(density * total_num_cells) for density in density_values]

""" phi_lists = []
for Ntot in number_agents_list:
    ffca = FFCA_wrap(number_rows, number_cols, Ntot, spawn_rate=0.025,
                 conflict_resolution_rate=0, alpha=0.3, delta=0.3,
                 static_field_strength=2.5, dynamic_field_strength=5,
                 horizontal_bias=50)  
    phi_lists.append(lane_formation(number_rows, number_cols, Ntot, steps, ffca))

# Make a plot with all density values (phi values vs iterations)

plt.figure(figsize=(10, 6))
for i, phi_values in enumerate(phi_lists):
    plt.plot(range(steps), phi_values, linestyle='-', label=f'Density: {density_values[i]}')
plt.xlabel("Iterations", fontsize=14)
plt.ylabel("Order Parameter", fontsize=14)
plt.title("Order Parameter vs Iterations", fontsize=16)
plt.grid(True)
plt.legend()
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig("order_parameter_vs_density.pdf")
plt.show()

# Save values in a CSV file with density as column headers
data = np.array(phi_lists).T  # Transpose to align each density as a column
columns = [f"Density {density}" for density in density_values]

df = pd.DataFrame(data, columns=columns)
df.to_csv("order_parameter_vs_density.csv", index=False) """

# All densities test
densities = np.linspace(0.05, 0.25, 100)
phi_values = []

for density in densities:
    Ntot = int(density * total_num_cells)
    ffca = FFCA_wrap(number_rows, number_cols, Ntot, spawn_rate=0.025,
                 conflict_resolution_rate=0, alpha=0.3, delta=0.3,
                 static_field_strength=2.5, dynamic_field_strength=5,
                 horizontal_bias=50)  
    phi_list = lane_formation(number_rows, number_cols, Ntot, steps, ffca)
    # average of last 20 values
    phi_values.append(np.mean(phi_list[-20:]))

# Make a plot with all density values (phi values vs density)
plt.figure(figsize=(10, 6))
plt.plot(densities, phi_values, linestyle='-', label='Order Parameter')
plt.xlabel("Density", fontsize=14)
plt.ylabel("Order Parameter", fontsize=14)
plt.title("Order Parameter vs Density", fontsize=16)
plt.grid(True)
plt.legend()
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig("order_parameter_vs_density_all.pdf")
plt.show()

# Save values in a CSV file
df = pd.DataFrame({'Density': densities, 'Order Parameter': phi_values})
df.to_csv("order_parameter_vs_density_all.csv", index=False)

# Different conflict resolution rates
conflict_resolution_rates = [0, 0.25, 0.5, 0.75, 1]
phi_lists = []

for rate in conflict_resolution_rates:
    ffca = FFCA_wrap(number_rows, number_cols, number_agents, spawn_rate=0.025,
                 conflict_resolution_rate=rate, alpha=0.3, delta=0.3,
                 static_field_strength=2.5, dynamic_field_strength=5,
                 horizontal_bias=50)  
    phi_lists.append(lane_formation(number_rows, number_cols, number_agents, steps, ffca))

# Make a plot with all conflict resolution rates (phi values vs iterations)
plt.figure(figsize=(10, 6))
for i, phi_values in enumerate(phi_lists):
    plt.plot(range(steps), phi_values, linestyle='-', label=f'Conflict Resolution Rate: {conflict_resolution_rates[i]}')
plt.xlabel("Iterations", fontsize=14)
plt.ylabel("Order Parameter", fontsize=14)
plt.title("Order Parameter vs Iterations", fontsize=16)
plt.grid(True)
plt.legend()
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig("order_parameter_vs_conflict_resolution_rate.pdf")
plt.show()

# Save values in a CSV file with conflict resolution rate as column headers
data = np.array(phi_lists).T  # Transpose to align each rate as a column
columns = [f"Conflict Resolution Rate {rate}" for rate in conflict_resolution_rates]

df = pd.DataFrame(data, columns=columns)
df.to_csv("order_parameter_vs_conflict_resolution_rate.csv", index=False)



    
