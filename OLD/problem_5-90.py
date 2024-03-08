import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Domain specifications
x_length = 0.09  # Length in meters
y_length = 0.045
delta_x = 0.015  # Grid spacing in meters
xnodes = int(x_length / delta_x) + 1
ynodes = int(y_length / delta_x) + 1

# Time stepping
delta_t = 0.1  # Time step size in seconds
total_time = 3600  # Total simulation time in seconds
time_steps = int(total_time / delta_t)

# Material properties
kappa = np.array([401, 90.7, 15.1])  # Thermal conductivities: Cu, Ni, Fe
rho = np.array([8960, 8908, 8000])  # Densities: Cu, Ni, Fe
Cp = np.array([385, 445, 500])  # Specific heat capacities: Cu, Ni, Fe

# Initial and boundary conditions
T_initial = 773  # Initial temperature in Kelvin
T_amb = 288  # Ambient temperature in Kelvin
T_bot = np.array([600, 700, 800, 900, 800, 700, 600]) + 273  # Boundary condition at the bottom in Kelvin

# Initialize the temperature array
T = np.full((time_steps, ynodes, xnodes), T_initial, dtype=np.float32)

# Material map
material_map = np.zeros((ynodes, xnodes), dtype=int)  # Entire domain as copper

# Mesh grid for plotting
x = np.linspace(0, x_length, xnodes)
y = np.linspace(0, y_length, ynodes)
X, Y = np.meshgrid(x, y)

# Simulation loop
for t in range(time_steps - 1):
    # Reapply the boundary condition at every time step for the bottom
    T[t, 0, :] = T_bot

    for i in range(1, ynodes - 1):
        for j in range(1, xnodes - 1):
            mat = material_map[i, j]  # Material at current position
            alpha = kappa[mat] / (rho[mat] * Cp[mat])
            T[t+1, i, j] = T[t, i, j] + alpha * delta_t * (
                (T[t, i+1, j] - 2*T[t, i, j] + T[t, i-1, j]) / delta_x**2 +
                (T[t, i, j+1] - 2*T[t, i, j] + T[t, i, j-1]) / delta_x**2)

# Animation and plotting setup
fig, ax = plt.subplots(figsize=(8, 4))
contour = plt.contourf(X, Y, T[0, :, :], 50, cmap='jet', vmin=T_initial, vmax=np.max(T_bot))

time_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, color='white', fontsize=12)
colorbar = fig.colorbar(contour, ax=ax)
colorbar.set_label('Temperature (K)')
def init():
    contour = plt.contourf(X, Y, T[0, :, :], 50, cmap='jet', vmin=T_initial, vmax=np.max(T_bot))
    time_text.set_text('')  # Initialize the text
    return contour.collections

def update(frame):
    ax.clear()
    contour = plt.contourf(X, Y, T[frame, :, :], 50, cmap='jet', vmin=T_initial, vmax=np.max(T_bot))
    # Update the timer text to the current simulation time
    current_time = frame * delta_t
    time_text.set_text(f'Time: {current_time:.1f} s')
    # Place the timer at the top left of the plot
    ax.text(0.05, 0.95, f'Time: {current_time:.1f} s', transform=ax.transAxes, color='white', fontsize=12)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Temperature Distribution Over Time')
    return contour.collections

# Reduce the total frames by selecting every nth frame
n = max(int(time_steps / 50), 1)  # Ensure at least 1 as a step size
anim = animation.FuncAnimation(fig, update, frames=100, init_func=init, blit=False)

plt.tight_layout()
plt.show()

#anim.save('5-90 transient.gif', writer='pillow', fps=10)
