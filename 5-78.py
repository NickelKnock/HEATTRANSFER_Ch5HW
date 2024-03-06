import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import cm
import sympy as sp

# Redefine the symbols without assumptions for clarity
T = sp.symbols('T1:9')  # Redefine symbols for T1 to T8

# Given values
k_val = 23  # thermal conductivity in W/m-K
Q_val = 8000  # heater power in W
DELTA_val = 0.1  # delta in m
w_val = 0.5  # width in m
L_val = 5  # length in m

# Material properties
C_p = 390  # J/kg-K
rho = 8900  # kg/m^3
k = 23  # W/m-K

# Redefine the equations based on the provided equations
equations = [
    (Q_val/(L_val*w_val))*(2*DELTA_val/k_val) + (T[4] + 2*T[1] - 4*T[0]),  # Node 1 (heat flux with heater)
    (Q_val/(L_val*w_val))*(2*DELTA_val/k_val) + (T[0] + 2*T[5] - 3*T[4]),  # Node 5 (heat flux with heater)
    0 + T[0] + T[2] + T[5] - 4*T[1],                                       # Node 2 (interior)
    0 + T[1] + T[3] + T[6] - 4*T[2],                                       # Node 3 (interior)
    0 + 2*T[2] + T[7] - 4*T[3],                                            # Node 4 (insulation)
    T[1] + T[4] + T[5] + T[6] - 4*T[5],                                    # Node 6 (interior)
    T[2] + T[5] + T[6] + T[7] - 4*T[6],                                    # Node 7 (interior)
    T[3] + 2*T[6] + T[7] - 4*T[7]                                          # Node 8 (insulation)
]

# Solve the system of equations
solutions = sp.solve(equations, T)

# Output the solutions in a readable format
solutions_clean = {str(t): solutions[t].evalf() for t in T}
print('Temps are as follows:')
for key, value in solutions_clean.items():
    print(f"{key}: {value:.4f}°C")

# Constants and initial conditions for the transient simulation
k_material = 23  # Thermal conductivity, W/m-K
C_p = 390  # Specific heat capacity, J/kg-K
rho = 8900  # Density, kg/m^3
Q_heater = 8000  # Heater power, W
delta_x = delta_y = 0.05  # Grid spacing, m
dt = 1  # Time step, s
total_time = 1000  # Total time, s

# Dimensions
length = 5.0  # Length of the block, m
height = 0.3  # Height of the block, m

# Nodes in the grid
nodes_x = int(length / delta_x) + 1  # Nodes in x
nodes_y = int(height / delta_y) + 1  # Nodes in y

# Initialize the temperature grid
T_grid = np.ones((nodes_y, nodes_x)) * 20  # Initial temperature of 20°C everywhere

# Apply the heater temperature
T_grid[0, :] = 500  # 500°C at the top due to the heater

# Simulation function
def update_temperatures(T, dt, C_p, rho, k, Q, dx, dy, nodes_x, nodes_y):
    T_new = np.copy(T)
    Q_vol = Q / (dx * dy * length)  # Distributing Q over the volume

    for j in range(1, nodes_y - 1):
        for i in range(1, nodes_x - 1):
            T_new[j, i] = T[j, i] + (k * dt / (C_p * rho * dx ** 2)) * (T[j, i + 1] - 2 * T[j, i] + T[j, i - 1]) \
                          + (k * dt / (C_p * rho * dy ** 2)) * (T[j + 1, i] - 2 * T[j, i] + T[j - 1, i]) \
                          + Q_vol * dt / (rho * C_p)

    # Boundary conditions
    T_new[0, :] = 500  # Heater boundary
    T_new[-1, :] = T_new[-2, :]  # Insulated bottom boundary
    T_new[:, 0] = T_new[:, -1] = 0  # Sides in contact with 0°C water

    return T_new

# Prepare storage for the temperature data over time
T_over_time = np.zeros((total_time, nodes_y, nodes_x))
T_over_time[0] = T_grid

# Main simulation loop
for t in range(1, total_time):
    T_grid = update_temperatures(T_grid, dt, C_p, rho, k_material, Q_heater, delta_x, delta_y, nodes_x, nodes_y)
    T_over_time[t] = T_grid

# Visualization with dual-plot (steady-state and transient animation)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Convert steady-state solutions to a 2D array for plotting (assuming symmetry for simplicity)
steady_state_values = np.array([float(value) for value in solutions_clean.values()]).reshape((2, 4))
steady_state_full = np.vstack([steady_state_values, np.flipud(steady_state_values)])  # Mirror for full grid

# Plot the steady-state temperature distribution
im1 = ax1.imshow(steady_state_full, cmap='hot', origin='lower', extent=(0, length, 0, height))
fig.colorbar(im1, ax=ax1)
ax1.set_title('Steady-state Temperature Distribution')
ax1.set_xlabel('Length (m)')
ax1.set_ylabel('Height (m)')

# Set up the transient temperature distribution heatmap
im2 = ax2.imshow(T_over_time[0], cmap='hot', origin='lower', aspect='auto', extent=(0, length, 0, height))
fig.colorbar(im2, ax=ax2)
ax2.set_title("Transient Temperature Distribution at t = 0 seconds")
ax2.set_xlabel('Length (m)')
ax2.set_ylabel('Height (m)')

# Animation update function
def animate(i):
    im2.set_data(T_over_time[i])
    ax2.set_title(f"Transient Temperature Distribution at t = {i*dt} seconds")
    return im2,

# Create the animation object
anim = FuncAnimation(fig, animate, frames=total_time, interval=50, blit=False, repeat=False)

plt.tight_layout()
plt.show()

# If needed, uncomment the line below to save the animation to a file.
#anim.save('heat_transfer_simulation.mp4', writer='ffmpeg', fps=20)
#anim.save('heat_transfer_simulation.gif', writer='pillow', fps=20)