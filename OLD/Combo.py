import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

"""
This program models the temperature distribution within a two-dimensional rectangular domain, solving for both 
steady-state and transient conditions using the finite difference method (FDM). It particularly accounts for various
 boundary conditions, including heat input, convection, and insulation, by applying the heat conduction equation and 
 simplifying it based on the physical setup.

Derivation of Boundary Conditions:

1. Middle (Heat Input):
- Assuming heat flux (qmid) in the z-direction and equal grid spacing (delta_x = delta_y), the heat balance equation at
 the middle boundary is simplified to:
    T0 = (qmid*delta_x + k_Cu*(Tleft + (Tbot + Ttop)/2) + k_Fe*(Tright + (Tbot + Ttop)/2)) / (2*(k_Cu + k_Fe)),
  where k_Cu and k_Fe are thermal conductivities of copper and iron, respectively, and T0, Tleft, Tright, Tbot, Ttop
   represent the temperatures at the center, left, right, bottom, and top nodes.

2. Top (Convection):
- For the top boundary, where convection is present, the temperature at the top boundary (T0) considering convection to
 the ambient (Tamb) is given by:
    T0 = (h*delta_x/k*(Tamb) + (Tleft + Tright)/2 + Tbot) / (h*delta_x/k + 2),
  with h as the convective heat transfer coefficient and k as the average thermal conductivity between k_Cu and k_Fe.

3. Left and Right (Insulation):
- Insulated boundaries imply no net heat flow across the boundary. This condition modifies the temperature equation for
 edge nodes adjacent to insulated boundaries by eliminating the contribution from the outside domain and adjusting contributions from the top and bottom by a factor of 2.

Implementation Notes:
- These derived equations are applied within a for-loop to iteratively solve for the temperature distribution, adjusting
 for material properties and boundary effects.
- The steady-state solution focuses on achieving equilibrium where internal heat generation/absorption balances with
 heat loss/gain at boundaries.
- The transient analysis extends these principles over time, updating temperatures based on previous states,
 diffusivity, and boundary interactions, using an explicit time-marching scheme.

Visualization of the temperature distribution provides insights into the effects of boundary conditions, material
 properties, and internal heat generation on thermal behavior within the domain.
"""


# Constants for the steady-state solution
m_l = 0.09  # m
n_l = 0.045  # m
kCu = 401  # W/m-K
kNi = 90.7
kStnls = 15.1
h = 125  # W/m^2-K
delta_x = 0.015  # 1.5 cm
# (m+1), (n+1)
normX = round(m_l / delta_x + 1)
normY = round(n_l / delta_x + 1)

# Initialize solution: the grid of u(k, i, j)
T_steady = np.empty((normY, normX))
T_ini = 20 + 273.15  # Initial condition everywhere inside the grid
T_amb = 15 + 273.15  # temperature in K
# Bottom boundary conditions + convert to Kelvin
T_bot = np.array([600, 700, 800, 900, 800, 700, 600]) + 273

# Set the initial condition
T_steady.fill(T_ini)
T_steady[0, :] = T_bot  # fixed T on bottom

maxChange = 1
tempT = np.empty((normY, normX))
iter_to_solve = 0

# Perform iterations until the change in temperature is below the threshold
while maxChange >= 0.001:
    iter_to_solve += 1
    tempT[:, :] = T_steady.copy()

    for i in range(1, normY):
        for j in range(normX):
            # Assign thermal conductivity based on position
            if 1 <= i:
                if 1 < j <= 3:
                    K = kCu  # Copper region
                elif 3 < j <= 5:
                    K = kNi  # Nickel region
                else:
                    K = kStnls  # Stainless steel region

            # Right boundary condition
            if j == normX - 1 and i < normY - 1:
                T_steady[i][j] = (T_steady[i + 1][j] + T_steady[i - 1][j] + T_steady[i][j - 1]) / 3

            # Left boundary condition
            elif j == 0 and i < normY - 1:
                T_steady[i][j] = (T_steady[i + 1][j] + T_steady[i - 1][j] + T_steady[i][j + 1]) / 3

            # Top convection (excluding top right corner)
            elif i == normY - 1 and j < normX - 1:
                T_steady[i][j] = ((h * delta_x / K * T_amb) + T_steady[i - 1][j] + (T_steady[i][j + 1] + T_steady[i][j - 1]) / 2) / \
                          (h * delta_x / K + 2)

            # Top right corner
            elif i == normY - 1 and j == normX - 1:
                T_steady[i][j] = ((h * delta_x / K * T_amb) + T_steady[i - 1][j] + T_steady[i][j - 1]) / \
                          (h * delta_x / K + 2)

            # Bulk region
            else:
                T_steady[i][j] = (T_steady[i + 1][j] + T_steady[i - 1][j] + T_steady[i][j + 1] + T_steady[i][j - 1]) / 4

    deltaT = abs(T_steady - tempT)
    maxChange = np.max(deltaT)
    print("Iteration: ", iter_to_solve, "Max delta T: ", maxChange)

print("Maximum Temperature : ", np.max(T_steady))
print(f"Steady-state solution took {iter_to_solve} iterations to complete")

# Convert to Celsius
T_steady -= 273

# Domain specifications for the transient solution
x_length, y_length = 0.09, 0.045  # Length in meters
delta_x = 0.015  # Grid spacing in meters
xnodes = int(x_length / delta_x) + 1
ynodes = int(y_length / delta_x) + 1

# Time stepping
delta_t = .1  # Time step size in seconds
total_time = 3600  # Total simulation time in seconds
time_steps = 100 #int(total_time / delta_t)

# Material properties (Thermal conductivities, Densities, Specific heat capacities)
kappa = np.array([401, 90.7, 15.1])  # Cu, Ni, Fe
rho = np.array([8960, 8908, 8000])   # Cu, Ni, Fe
Cp = np.array([385, 445, 500])       # Cu, Ni, Fe

# Initial and boundary conditions
T_initial = 773  # Initial temperature in Kelvin
T_amb = 288  # Ambient temperature in Kelvin
T_bot = np.array([600, 700, 800, 900, 800, 700, 600]) + 273  # Bottom boundary in Kelvin

# Initialize the temperature array
T_transient = np.full((time_steps, ynodes, xnodes), T_initial, dtype=np.float32)

# Material map (0 for Cu initially)
material_map = np.zeros((ynodes, xnodes), dtype=int)

# Mesh grid for plotting for transient solution
x = np.linspace(0, x_length, xnodes)
y = np.linspace(0, y_length, ynodes)
X, Y = np.meshgrid(x, y)

# Simulation loop for the transient solution
for t in range(time_steps - 1):
    T_transient[t, 0, :] = T_bot  # Reapply bottom boundary condition

    for i in range(1, ynodes - 1):
        for j in range(1, xnodes - 1):
            mat = material_map[i, j]
            alpha = kappa[mat] / (rho[mat] * Cp[mat])
            # Update temperatures based on diffusion equation
            T_transient[t+1, i, j] = T_transient[t, i, j] + alpha * delta_t * (
                (T_transient[t, i+1, j] - 2*T_transient[t, i, j] + T_transient[t, i-1, j]) / delta_x**2 +
                (T_transient[t, i, j+1] - 2*T_transient[t, i, j] + T_transient[t, i, j-1]) / delta_x**2)

# Animation and plotting setup
fig, axs = plt.subplots(1, 2, figsize=(12, 4))

# Plot steady-state solution on the left side
axs[0].set_title("Steady-State Temperature (C)")
axs[0].set_xlabel("x (m)")
axs[0].set_ylabel("y (m)")
contour_steady = axs[0].pcolormesh(X, Y, T_steady, cmap=plt.cm.jet, shading='gouraud')
colorbar_steady = fig.colorbar(contour_steady, ax=axs[0], label='Temperature (C)')

# Plot transient solution on the right side
axs[1].set_title("Transient Temperature (C)")
axs[1].set_xlabel("x (m)")
axs[1].set_ylabel("y (m)")
contour_transient = axs[1].contourf(X, Y, T_transient[0, :, :], 50, cmap='jet', vmin=T_initial, vmax=np.max(T_bot))
colorbar_transient = fig.colorbar(contour_transient, ax=axs[1], label='Temperature (C)')

time_text = axs[1].text(0.05, 0.95, '', transform=axs[1].transAxes, color='white', fontsize=12)


# Function to update the transient solution during animation
def update(frame):
    contour_transient = axs[1].contourf(X, Y, T_transient[frame, :, :], 50, cmap='jet', vmin=T_initial,
                                         vmax=np.max(T_bot))

    # Update the time text in the transient subplot
    current_time = frame * delta_t
    time_text.set_text(f'Time: {current_time:.1f} s')


# Call FuncAnimation to update the transient solution in the transient subplot
anim = animation.FuncAnimation(fig, update, frames=time_steps, blit=False)
anim.save('5-90 combo.gif', writer='pillow', fps=1)
plt.show()
