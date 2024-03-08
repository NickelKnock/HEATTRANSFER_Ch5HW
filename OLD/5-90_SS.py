import numpy as np
import matplotlib.pyplot as plt

print("2D heat equation solver")

x_length = 0.09 #9 cm by 4.5 cm
y_length = 0.045

# max_iters = 100 #maximum number of iterations on solver


kappa_Cu = 401 # W/m-K
kappa_Ni = 90.7
kappa_Fe = 15.1
h = 125 # W/m^2-K
delta_x = 0.015 # 1.5 cm
xnorm = x_length/delta_x + 1 #node on each end need to add 1
ynorm = y_length/delta_x + 1
xnorm = round(xnorm)
ynorm = round(ynorm)
print("nodes in x: ", xnorm)
print("nodes in y: ", ynorm)

# Initialize solution: the grid of u(k, i, j)
T = np.empty((ynorm, xnorm))

# Initial condition everywhere inside the grid
T_initial = 293

# Boundary conditions

T_amb = 288 #temperature in K
T_bot = np.array([600, 700, 800, 900, 800, 700, 600])
T_bot = np.add(T_bot,273)

# Set the initial condition
T.fill(T_initial)

# Set the boundary conditions
# middle is heat in, assume z direction = 1,  so:
#  qmid*deltay + k_Cu*deltay*(Tleft-T0)/deltax + k_Fe*deltay*(Tright-T0)/deltax +
#   (k_Cu +k_Fe)*deltax/2*(Tbot-T0)/deltay + (k_Cu +k_Fe)*deltax/2*(Ttop-T0)/deltay
#  == 0
# because dx = dy this simplifies to:
#   0= [qmid*deltax + k_Cu*(Tleft-T0) + k_Fe*(Tright-T0) +
#   (k_Cu +k_Fe)/2*(Tbot-T0) + (k_Cu +k_Fe)/2*(Ttop-T0)

#which becomes:
    # 0 = qmid*deltax + k_Cu * (Tleft + Tbot/2 + Ttop/2) 
    # + k_Fe * (Tright + Tbot/2 + Ttop/2) - 2 * (k_Cu + k_Fe) * T0
# So
    # T0 = (qmid*deltax + k_Cu * (Tleft + Tbot/2 + Ttop/2) 
    # + k_Fe * (Tright + Tbot/2 + Ttop/2)) / (2 * (k_Cu + k_Fe))
# This equation will be used in the for loop

#Top is convection:
    # h * deltax * (Tamb - T0) + k* deltay/2 * (Tleft - T0)/deltax
    # + k * deltay/2 * (Tright - T0)/deltax + k* deltax * (Tbot - T0)/deltay = 0
#solve  for T0
    # T0 = (h * deltax /k * (Tamb) + Tleft/2
    # + Tright/2 + Tbot) /(h *deltax/k + 2)
#Note that the top left and right corners will be technically different. We will make top 
# right different out of necessity but will assume top left is same equation...
    
#Left and Right are insulation
    # use same as bulk but delete contribution from outside domain and 
    # divide top and bottom by 2
T[:1, :] = T_bot #fixed T on bottom

maxTchange = 1
tempT = np.empty((ynorm, xnorm))
iter_to_solve = 0

while(maxTchange>=0.001):
#for k in range(max_iters):
    iter_to_solve = iter_to_solve + 1
    tempT[:, :] = T.copy()
    for i in range(1, ynorm, 1):
        for j in range(0, xnorm, 1):
            if i >= 1 and j > 1 and j <= 3:
                kappa = kappa_Cu
            elif i >= 1 and j > 3 and j <= 5:
                kappa = kappa_Ni
            else:
                kappa = kappa_Fe
            if j == xnorm-1 and i < ynorm-1: #right boundary described above
                T[i][j] = (T[i+1][j]/2 + T[i-1][j]/2 + T[i][j-1])/2 
            elif j == 0 and i < ynorm-1: #left boundary described above
                T[i][j] = (T[i+1][j]/2 + T[i-1][j]/2 + T[i][j+1])/2 
            elif i == ynorm-1 and j < xnorm-1: #top convection from above
                T[i][j] = ((h * delta_x / kappa * T_amb) + T[i-1][j] + T[i][j+1]/2 + T[i][j-1]/2) / (h * delta_x / kappa + 2)
            elif i == ynorm-1 and j == xnorm-1: #corner top
                T[i][j] = ((h * delta_x / kappa * T_amb) + T[i-1][j] + T[i][j-1]) / (h * delta_x / kappa + 2)
            else: #bulk
                T[i][j] = (T[i+1][j] + T[i-1][j] + T[i][j+1] + T[i][j-1])/4
    deltaT = np.subtract(T,tempT[:, :])
    deltaT = abs(deltaT)
    maxTchange = np.max(deltaT)
    print("Max delta T: ", maxTchange)
print("Maximum Temperature : ", np.max(T))
print(f"Solution took {iter_to_solve:.0f} iterations to complete")    
#convert to K from C
T_convert = np.empty((ynorm, xnorm))
T_convert.fill(273)
T = np.subtract(T,T_convert)
plt.clf()

plt.title("Temperature (C)")
plt.xlabel("x")
plt.ylabel("y")

# This is to plot u_k (u at time-step k)
xrange = np.mgrid[slice(0, x_length+delta_x, delta_x)]
yrange = np.mgrid[slice(0, y_length+delta_x, delta_x)]
plt.pcolormesh(xrange, yrange, T, cmap=plt.cm.jet, vmin=round(np.min(T)), vmax=round(np.max(T)))
plt.colorbar()
plt.savefig('5-90_SS.png')

print("Done!")