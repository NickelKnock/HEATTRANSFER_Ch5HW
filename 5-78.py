import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import sympy as sp

"""
This code models the temperature distribution within a two-dimensional block subject to heating. It computes both the
 steady-state and transient temperature distributions using the finite difference method (FDM).

Steady-State Analysis:
- The steady-state condition assumes no time-dependent changes in temperature. 
- The Laplace equation for steady-state heat conduction (without internal heat generation) in two dimensions is used:
 ∇²T = 0, where T is temperature.
- For discrete nodes within the material, the equation is approximated using the finite difference approach as:
  T[i,j] = (T[i+1,j] + T[i-1,j] + T[i,j+1] + T[i,j-1]) / 4
- This formula is applied iteratively to interior nodes, with boundary conditions applied at the edges and corners of
 the domain. (See the code itself for the final algebraic expressions) 
- Specific heat source/sink conditions are incorporated to simulate effects like heating and insulation.

Transient Analysis:
- The transient analysis accounts for time-dependent changes in temperature within the block.
- The heat equation for transient heat conduction is: ρC_p(∂T/∂t) = k∇²T + Q, where ρ is density, C_p is specific heat
 capacity, k is thermal conductivity, and Q is the rate of internal heat generation.
- In discrete form, using the forward time, centered space (FTCS) method for the time derivative and central difference
 for spatial derivatives, the equation becomes:
  T_new[i,j] = T[i,j] + (dt * k / (ρC_pdx²)) * (T[i+1,j] - 2T[i,j] + T[i-1,j]) +
   (dt * k / (ρC_pdy²)) * (T[i,j+1] - 2T[i,j] + T[i,j-1]) + (Q_vol * dt / (ρC_p))
  
- Here, dx and dy are the spatial discretization in the x and y directions, respectively, dt is the time step, and Q_vol
 represents volumetric heating (distributed evenly across the domain if applicable).

Both analyses are implemented in the code, where:
- Steady-state conditions are solved using symbolic algebra to solve the system of equations representing node
 temperatures.
- Transient conditions are simulated over time using an explicit time-marching scheme, updating the temperature of each
 node based on its previous state and the states of its neighbors.

The results are visualized to show the temperature distribution, highlighting the effects of boundary conditions,
 internal heat generation, and material properties.
"""

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
    (Q_val / (L_val * w_val)) * (2 * DELTA_val / k_val) + (T[4] + 2 * T[1] - 4 * T[0]),
    # Node 1 (heat flux with heater)
    (Q_val / (L_val * w_val)) * (2 * DELTA_val / k_val) + (T[0] + 2 * T[5] - 3 * T[4]),
    # Node 5 (heat flux with heater)
    0 + T[0] + T[2] + T[5] - 4 * T[1],  # Node 2 (interior)
    0 + T[1] + T[3] + T[6] - 4 * T[2],  # Node 3 (interior)
    0 + 2 * T[2] + T[7] - 4 * T[3],  # Node 4 (insulation)
    T[1] + T[4] + T[5] + T[6] - 4 * T[5],  # Node 6 (interior)
    T[2] + T[5] + T[6] + T[7] - 4 * T[6],  # Node 7 (interior)
    T[3] + 2 * T[6] + T[7] - 4 * T[7]  # Node 8 (insulation)
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

# Correct the color distribution by transposing the array
steady_state_colors_corrected = steady_state_full.T

# Plot the color-corrected steady-state temperature distribution on the left graph
im1 = ax1.imshow(steady_state_colors_corrected, cmap='hot', origin='upper', aspect='auto',
                 extent=(0, length, 0, height))
fig.colorbar(im1, ax=ax1)
ax1.set_title('Steady-state Temperature Distribution')
ax1.set_xlabel('Length')  # Adjusted to reflect actual units correctly
ax1.set_ylabel('Height')  # Adjusted to reflect actual units correctly

vmin_temp = 50  # Minimum temperature value for color mapping
vmax_temp = 500  # Maximum temperature value for color mapping

# Set up the transient temperature distribution heatmap with adjusted color contrast
im2 = ax2.imshow(T_over_time[0], cmap='hot', origin='upper', aspect='auto', extent=(0, length, 0, height),
                 vmin=vmin_temp, vmax=vmax_temp)
fig.colorbar(im2, ax=ax2)
ax2.set_title("Transient Temperature Distribution at t = 0 seconds")
ax2.set_xlabel('Length (m)')
ax2.set_ylabel('Height (m)')  # Adjusted to reflect actual units correctly


# Animation update function
def animate(i):
    im2.set_data(T_over_time[i])
    ax2.set_title(f"Transient Temperature Distribution at t = {i * dt} seconds")
    return im2,


# Create the animation object
anim = FuncAnimation(fig, animate, frames=total_time, interval=50, blit=False, repeat=False)

plt.tight_layout()
plt.show()

# If needed, uncomment the line below to save the animation to a file.
# anim.save('heat_transfer_simulation.mp4', writer='ffmpeg', fps=20)
anim.save('heat_transfer_simulation.gif', writer='pillow', fps=20)
