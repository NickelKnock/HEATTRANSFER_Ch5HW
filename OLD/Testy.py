import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Domain specifications
x_length, y_length = 0.09, 0.045  # Length in meters
delta_x = 0.015  # Grid spacing in meters
xnodes = int(x_length / delta_x) + 1
ynodes = int(y_length / delta_x) + 1

# Time stepping
delta_t = 0.1  # Time step size in seconds
total_time = 3600  # Total simulation time in seconds
time_steps = int(total_time / delta_t)

# Material properties (Thermal conductivities, Densities, Specific heat capacities)
kappa = np.array([401, 90.7, 15.1])  # Cu, Ni, Fe
rho = np.array([8960, 8908, 8000])   # Cu, Ni, Fe
Cp = np.array([385, 445, 500])       # Cu, Ni, Fe

# Initial and boundary conditions
T_initial = 773  # Initial temperature in Kelvin
T_amb = 288  # Ambient temperature in Kelvin
T_bot = np.array([600, 700, 800, 900, 800, 700, 600]) + 273  # Bottom boundary in Kelvin

# Initialize the temperature array
T = np.full((time_steps, ynodes, xnodes), T_initial, dtype=np.float32)

# Material map (0 for Cu initially)
material_map = np.zeros((ynodes, xnodes), dtype=int)

# Mesh grid for plotting
x = np.linspace(0, x_length, xnodes)
y = np.linspace(0, y_length, ynodes)
X, Y = np.meshgrid(x, y)

# Simulation loop
for t in range(time_steps - 1):
    T[t, 0, :] = T_bot  # Reapply bottom boundary condition
    for i in range(1, ynodes - 1):
        for j in range(1, xnodes - 1):
            mat = material_map[i, j]
            alpha = kappa[mat] / (rho[mat] * Cp[mat])
            # Update temperatures based on diffusion equation
            T[t+1, i, j] = T[t, i, j] + alpha * delta_t * (
                (T[t, i+1, j] - 2*T[t, i, j] + T[t, i-1, j]) / delta_x**2 +
                (T[t, i, j+1] - 2*T[t, i, j] + T[t, i, j-1]) / delta_x**2)

# Animation setup
fig, ax = plt.subplots(figsize=(8, 4))
def init():
    ax.clear()
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title('Temperature Distribution Over Time')
    return plt.contourf(X, Y, T[0, :, :], 50, cmap='jet', vmin=T_initial, vmax=np.max(T_bot))

def update(frame):
    ax.clear()
    plt.contourf(X, Y, T[frame, :, :], 50, cmap='jet', vmin=T_initial, vmax=np.max(T_bot))
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.title(f'Temperature Distribution at Time: {frame * delta_t:.1f} s')
    return []

# Adjusted for slower animation: increased interval and larger step size for frames
frame_step = max(int(time_steps/10 ), 1)  # Adjust step size for fewer frames
anim = animation.FuncAnimation(fig, update, frames=np.arange(0, time_steps, frame_step), init_func=init, interval=200, blit=False)  # Increased interval to 200 ms

plt.colorbar(init(), ax=ax).set_label('Temperature (K)')
plt.tight_layout()
plt.show()

# Uncomment below to save the animation with a slower frame rate
# anim.save('temperature_distribution_slow.gif', writer='pillow', fps=5)