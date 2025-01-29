from FFCA_wrap import FFCA_wrap
from Grid import Grid, Pos
from metrics import order_parameter, mean_order_parameter
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.cm as cm
import matplotlib.colors as mcolors


def lane_formation(rows, cols, Ntot, steps, ffca):
    # Calculate the mean phi for a random distribution of agents
    random_phi_values = []
    for _ in range(500):
        ffca_random = FFCA_wrap(rows, cols, Ntot)
        N1, N2 = ffca.agents_in_row(ffca_random.structure)
        random_phi_values.append(order_parameter(Ntot, N1, N2))
    phi_zero = np.mean(random_phi_values)

    phi_values = np.zeros(steps)

    for i in range(steps):
        current_phi = order_parameter(Ntot, *ffca.agents_in_row(ffca.structure))
        current_mean_phi = mean_order_parameter(current_phi, phi_zero)
        phi_values[i] = current_mean_phi
        ffca.step()
    return phi_values


steps = 200
number_agents = 50
number_rows = 80
number_cols = 20

# Define horizontal bias values
horizontal_bias_values = [1, 25, 50, 100, 250, 500, 1000, 5000]

# Colourmap
cmap = plt.colormaps.get_cmap("Blues")  
new_cmap = mcolors.ListedColormap(cmap(np.linspace(0.4, 1, len(horizontal_bias_values))))  

# Normalize to map values correctly
norm = mcolors.Normalize(vmin=min(horizontal_bias_values), vmax=max(horizontal_bias_values))

# Run simulations
phi_lists = []
for bias in horizontal_bias_values:
    print(f"Horizontal Bias: {bias}")
    ffca = FFCA_wrap(number_rows, number_cols, number_agents, spawn_rate=0.025,
                     conflict_resolution_rate=0, alpha=0.3, delta=0.3,
                     static_field_strength=2.5, dynamic_field_strength=3,
                     horizontal_bias=bias)  
    phi_lists.append(lane_formation(number_rows, number_cols, number_agents, steps, ffca))

# Plot results
fig, ax = plt.subplots(figsize=(10, 6))

for i, phi_values in enumerate(phi_lists):
    color = new_cmap(i / (len(horizontal_bias_values) - 1))  # Get a color from the trimmed colormap
    ax.plot(range(steps), phi_values, linestyle='-', 
            color=color, linewidth=2, alpha=0.8, 
            label=f'Bias: {horizontal_bias_values[i]}')

# Add colorbar to indicate bias levels
sm = cm.ScalarMappable(cmap=new_cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label("Horizontal Bias", fontsize=14)

ax.set_xlabel("Iterations", fontsize=14)
ax.set_ylabel("Lane Formation", fontsize=14)
ax.set_title("Lane Formation vs Iterations", fontsize=16)
ax.grid(True)
ax.legend()
ax.tick_params(axis='both', labelsize=12)
plt.tight_layout()
plt.savefig("lane_formation_vs_horizontal_bias_dark_blues.pdf")
plt.show() 

# Different density values
total_num_cells = number_rows * number_cols

density_values = [0.05, 0.1, 0.15, 0.2, 0.25]
number_agents_list = [int(density * total_num_cells) for density in density_values]

phi_lists = []
for Ntot in number_agents_list:
    print(f"Density: {Ntot / total_num_cells}")
    ffca = FFCA_wrap(number_rows, number_cols, Ntot, spawn_rate=0.025,
                 conflict_resolution_rate=0, alpha=0.3, delta=0.3,
                 static_field_strength=2.5, dynamic_field_strength=3,
                 horizontal_bias=5000)  
    phi_lists.append(lane_formation(number_rows, number_cols, Ntot, steps, ffca))

# Make a plot with all density values (phi values vs iterations)

plt.figure(figsize=(10, 6))
for i, phi_values in enumerate(phi_lists):
    plt.plot(range(steps), phi_values, linestyle='-', label=f'Density: {density_values[i]}')
plt.xlabel("Iterations", fontsize=14)
plt.ylabel("Lane formation", fontsize=14)
plt.title("Lane formation vs Iterations", fontsize=16)
plt.grid(True)
plt.legend()
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig("lane_formatiom_vs_density.pdf")
plt.show()


# Lets run the simulation 10 times to get the standard deviation of the order parameter

n_runs = 10
densities = np.linspace(0.05, 0.25, 10)
phi_values = []

for n in range(n_runs):
    print(f"Run number {n+1}")
    phi_values.append([])
    for density in densities:
        print(f"Density: {density}")
        Ntot = int(density * total_num_cells)
        ffca = FFCA_wrap(number_rows, number_cols, Ntot, spawn_rate=0.025,
                     conflict_resolution_rate=0, alpha=0.3, delta=0.3,
                     static_field_strength=2.5, dynamic_field_strength=3,
                     horizontal_bias=5000)  
        phi_list = lane_formation(number_rows, number_cols, Ntot, steps, ffca)
        phi_values[n].append(np.mean(phi_list[-100:]))  # average of last 100 values

phi_values = np.array(phi_values)

# Calculate the mean and standard deviation of the order parameter
mean_phi_values = np.mean(phi_values, axis=0)
std_phi_values = np.std(phi_values, axis=0)

# Make a plot with mean and standard deviation of the order parameter
plt.figure(figsize=(10, 6))
plt.plot(densities, mean_phi_values, linestyle='-', label='Mean Order Parameter')
# Do error bars with std
plt.errorbar(densities, mean_phi_values, yerr=std_phi_values, fmt='o', label='Standard Deviation')
plt.xlabel("Density", fontsize=14)
plt.ylabel("Lane formation", fontsize=14)
plt.title("Lane formation vs Density", fontsize=16)
plt.grid(True)
plt.legend()
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig("ane_formatiom_vs_density_all_2.pdf")
plt.show()

# Different conflict resolution rates
conflict_resolution_rates = [0, 0.25, 0.5, 0.75, 1]
phi_lists = []

for rate in conflict_resolution_rates:
    print(f"Conflict Resolution Rate: {rate}")
    ffca = FFCA_wrap(number_rows, number_cols, number_agents, spawn_rate=0.025,
                 conflict_resolution_rate=rate, alpha=0.3, delta=0.3,
                 static_field_strength=2.5, dynamic_field_strength=3,
                 horizontal_bias=5000)  
    phi_lists.append(lane_formation(number_rows, number_cols, number_agents, steps, ffca))

# Make a plot with all conflict resolution rates (phi values vs iterations)
plt.figure(figsize=(10, 6))
for i, phi_values in enumerate(phi_lists):
    plt.plot(range(steps), phi_values, linestyle='-', label=f'Conflict Resolution Rate: {conflict_resolution_rates[i]}')
plt.xlabel("Iterations", fontsize=14)
plt.ylabel("Lane formation", fontsize=14)
plt.title("Lane formation vs Conflict Resolution Rate", fontsize=16)
plt.grid(True)
plt.legend()
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig("oane_formation_vs_conflict_resolution_rate.pdf")
plt.show()


# Define corridor widths
length = 100
corridor_width = [20, 40, 60, 80, 100]
phi_lists = []

# Choose a colormap (e.g., "plasma" or "viridis")
cmap = plt.colormaps.get_cmap("viridis")  
norm = mcolors.Normalize(vmin=min(corridor_width), vmax=max(corridor_width))

# Run simulations
for width in corridor_width:
    print(f"Corridor Length: {length}")
    number_agents = int(0.05 * width * length)
    print(f"Number of agents: {number_agents}")
    
    ffca = FFCA_wrap(width, length, number_agents, spawn_rate=0.025,
                     conflict_resolution_rate=0, alpha=0.3, delta=0.3,
                     static_field_strength=2.5, dynamic_field_strength=3,
                     horizontal_bias=5000)  
    phi_lists.append(lane_formation(width, length, number_agents, steps, ffca))

# Plot results
fig, ax = plt.subplots(figsize=(10, 6))

for i, phi_values in enumerate(phi_lists):
    color = cmap(norm(corridor_width[i]))  # Assign color based on corridor width
    ax.plot(range(steps), phi_values, linestyle='-', 
            color=color, linewidth=2, alpha=0.8, 
            label=f'Length: {corridor_width[i]}')

# Add colorbar to indicate corridor width
sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax)  
cbar.set_label("Corridor Length", fontsize=14)

# Labels and formatting
ax.set_xlabel("Iterations", fontsize=14)
ax.set_ylabel("Lane Formation", fontsize=14)
ax.set_title("Lane Formation vs Corridor Width", fontsize=16)
ax.grid(True)
ax.legend()
ax.tick_params(axis='both', labelsize=12)
plt.tight_layout()
plt.savefig("lane_formation_vs_corridor_width.pdf")
plt.show() 
