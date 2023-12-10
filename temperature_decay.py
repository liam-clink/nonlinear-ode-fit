from sympy import symbols, lambdify
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

T = symbols("T")  # temperature (Kelvin)
t = symbols("t")  # time (seconds)
A, B, T0 = symbols("A B T0")  # coefficients
Tdot_full = -A * (T**4 - T0**4) - B * (T - T0)
Tdot_fourier = -B * (T - T0)
f_Tdot_full = lambdify((t, T, A, B, T0), Tdot_full)
f_Tdot_fourier = lambdify((t, T, A, B, T0), Tdot_fourier)

# Calculate coefficients (these will be changed to initial guesses in later version)
SB_const = 5.67e-8  # W m^-2 K^-4
volumetric_heat_capacity = 3.0e6  # J m^-3 K^-1
volume = 4.1e-6  # m^3
heat_capacity = volumetric_heat_capacity * volume  # J K^-1
area = 2.6e-3  # m^2
thermal_conductivity = 1.0e2  # W m^-1 K^-1
support_length = 0.1  # m
support_area = 1.0e-4  # m^2
thermal_conductance = thermal_conductivity * support_area / support_length  # W K^-1

tstart = 0.0
tend = 50.0  # seconds
times = np.linspace(tstart, tend, 100)
initial_temperature = 1000.0
coefficients = (SB_const * area / heat_capacity, thermal_conductance, 300.0)
print(coefficients)
full_solution = solve_ivp(
    f_Tdot_full, (tstart, tend), (initial_temperature,), t_eval=times, args=coefficients
)
fourier_solution = solve_ivp(
    f_Tdot_fourier,
    (tstart, tend),
    (initial_temperature,),
    t_eval=times,
    args=coefficients,
)

plt.plot(times, full_solution.y.T, label="Full")
plt.plot(times, fourier_solution.y.T, label="Fourier")
plt.legend()
plt.title("Temperature Decay")
plt.ylabel("Temperature (K)")
plt.xlabel("Time (s)")
plt.show()
