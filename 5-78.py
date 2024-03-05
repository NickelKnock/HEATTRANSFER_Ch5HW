import numpy as np
import sympy as sp

# Redefine the symbols without assumptions for clarity
T = sp.symbols('T1:9')  # Redefine symbols for T1 to T8

# Given values
k_val = 23  # thermal conductivity in W/m-K
Q_val = 8000  # heater power in W
DELTA_val = 0.1  # delta in m
w_val = 0.5  # width in m
L_val = 5  # length in m

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
solutions_clean


def transient_heat_conduction(T_initial, alpha, delta_t, delta_x, delta_y, time_steps, T_top, T_side, T_bottom):
    """
    Simulates transient heat conduction in a 2D block with specific boundary conditions.

    Parameters:
    T_initial (float): Initial uniform temperature of the block in °C.
    alpha (float): Thermal diffusivity of the material (m^2/s).
    delta_t (float): Time step size (s).
    delta_x, delta_y (float): Spatial discretization in x and y directions (m).
    time_steps (int): Number of time steps to simulate.
    T_top (float): Temperature at the top surface due to the heater in °C.
    T_side (float): Temperature at the sides in contact with iced water in °C.
    T_bottom (float): Temperature at the bottom (negligible heat transfer) in °C.

    Returns:
    np.array: Temperature distribution of the block at the final time step.
    """
    # Calculate the number of spatial points in each dimension
    nx, ny = int(L / delta_x) + 1, int(H / delta_y) + 1

    # Initialize the temperature grid
    T = np.full((nx, ny), T_initial)

    # Apply the initial boundary conditions
    T[:, -1] = T_top  # Top surface heated by the heater
    T[0, :] = T_side  # Left side in contact with iced water
    T[-1, :] = T_side  # Right side in contact with iced water
    T[:, 0] = T_bottom  # Bottom surface with negligible heat transfer

    # Time-stepping loop
    for _ in range(time_steps):
        T_new = np.copy(T)

        # Update temperature for each interior point
        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                T_new[i, j] = T[i, j] + alpha * delta_t * (
                        (T[i + 1, j] - 2 * T[i, j] + T[i - 1, j]) / delta_x ** 2 +
                        (T[i, j + 1] - 2 * T[i, j] + T[i, j - 1]) / delta_y ** 2
                )

        # Update the temperature grid
        T = T_new

        # Reapply boundary conditions for sides and top if necessary
        T[:, -1] = T_top  # Top surface heated by the heater
        T[0, :] = T_side  # Left side in contact with iced water
        T[-1, :] = T_side  # Right side in contact with iced water
        # Bottom boundary remains unchanged due to negligible heat transfer
    print(' New T(t):', T )
    return T
