import numpy as np
from scipy.integrate import solve_ivp
import scipy.optimize as opt
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

plt.rcParams["figure.figsize"] = np.array(plt.rcParams["figure.figsize"]) * 1.5

# Grav. konst.
kapa = 6.6743e-11

# Sonce
M_s = 1.989e30

# Zemlja
M_z = 5.9724e24
#r_zs = 146e9 # razdalja sonca do zemlje
r_zs = 147.095e9
#v_z = 29.78e3
v_z = 30.29e3

# Luna
M_l = 0.07346e24
r_min = 0.3633e9
r_max = 0.4055e9
v_min = (2 * kapa * M_z * (1/r_min - 1/r_max) / (1 - (r_min / r_max)**2)) ** 0.5
#v_min = 1082
theta = np.pi / 180 * 5.15 # Kot lunine orbite glede na ravnino zemljine orbite okoli sonca

def df_dt(t, f):
    r_zl = np.array(f[:3]) # vektor od zemlje do lune
    v_ls = np.array(f[3:6]) # hitrost lune glede na sonce
    r_sz = np.array(f[6:9]) # vektor od sonca do zemlje
    v_z = np.array(f[9:]) # hitrost zemlje glede na sonce
    r_sl = r_zl + r_sz # vektor od sonca do lune
    
    a_sz = kapa * M_s / (np.linalg.norm(r_sz) ** 3) * (-r_sz)
    a_sl = kapa * M_s / (np.linalg.norm(r_sl) ** 3) * (-r_sl)
    a_zl = kapa * M_z / (np.linalg.norm(r_zl) ** 3) * (-r_zl)
    a_ls = a_sl + a_zl

    return [v_ls[0] - v_z[0], v_ls[1] - v_z[1], v_ls[2] - v_z[2], a_ls[0], a_ls[1], a_ls[2], v_z[0], v_z[1], v_z[2], a_sz[0], a_sz[1], a_sz[2]]

def orbita(t, f):
    r_zl = np.array(f[:3]) # vektor od zemlje do lune
    return r_zl[0]

orbita.direction = 1

#t_end = 32_000_000
n_orbit = 100
t_end = 2_370_000 * n_orbit
t = np.linspace(0.0, t_end, 1000 * n_orbit)

#      L     Poz:           Hitrost:            S:   Poz:       Hitrost:
#          x                       y               z               v_x     v_y v_z   x   y    z
zac_pog = [0, r_min*np.cos(theta), r_min*np.sin(theta), v_min + v_z, 0, 0,   0, r_zs, 0, v_z, 0, 0]
#zac_pog = [r_min*np.cos(theta), 0, r_min*np.sin(theta), 0 + v_z, v_min, 0,   ]#0, r_zs, 0, v_z, 0, 0]

sol = solve_ivp(df_dt, [0, t_end], y0=zac_pog, t_eval=t, rtol=1e-7, atol=1e-7, method='DOP853', 
                events=[orbita])

#print(f"{sol.y[0][-1]}  {sol.y[1][-1]}")

fig = plt.figure()
 
# syntax for 3-D projection
ax = plt.axes(projection ='3d')

# Ravnina zemline orbite okoli sonca
X = np.array([-5e8, 5e8])
Y = np.array([-5e8, 5e8])
X, Y = np.meshgrid(X, Y)
Z = X*0 + Y*0


# plotting
# ax.plot3D(sol.y[6],sol.y[7], sol.y[8], 'g')
# ax.scatter(0, 0, 0, color='y') #sonce
# ax.plot3D(sol.y[0] + sol.y[6],sol.y[1] + sol.y[7], sol.y[2] + sol.y[8], 'r')

ax.plot3D(sol.y[0], sol.y[1], sol.y[2], label="Lunina pot", c='r', linewidth=0.1)
ax.scatter(0, 0, 0, color='g',label="Zemlja") # zemlja
# ax.scatter(0, r_min*np.cos(theta) + r_zs, r_min*np.sin(theta), color='b') 
ax.plot_surface(X, Y, Z, color='blue', label="Ravnina Zemljine orbite okoli Sonca", alpha=0.3)


ax.set_title('Pot Lune okoli Zemlje')
ax.set_xlabel("X [m]")
ax.set_ylabel("Y [m]")
ax.set_zlabel("Z [m]")
#ax.set_xlim(-5e8, 5e8)
#ax.set_ylim(-5e8, 5e8)
ax.set_zlim(-5e8, 5e8)
leg = plt.legend()
leg_lines =  leg.get_lines()
plt.setp(leg_lines, linewidth=4)
plt.show()

event_f = np.array(sol.y_events[0])

plt.plot(sol.y[0], sol.y[1], linewidth = 0.2, label="pot Lune")
plt.plot(event_f[:, 0], event_f[:, 1], 'o', c='y')
plt.plot(0, 0, 'o', c='g', label="Zemlja")
plt.title("projekcija Lunine poti na XY ravnino")
plt.xlabel("X [m]")
plt.ylabel("Y [m]")
leg = plt.legend()
leg_lines =  leg.get_lines()
plt.setp(leg_lines, linewidth=1)
plt.show()

ena_orbita = sol.t_events[0][-1] / len(sol.t_events[0][1:]) / (60 * 60 * 24)
orb = (np.roll(sol.t_events[0][:], -1 )- sol.t_events[0][:]) / (60 * 60 * 24)
orb[-1] = 0
orb = np.roll(orb, 1)
k = 29.53 / ena_orbita
print(f"t  orbite = {ena_orbita} dni")

event_f = np.array(sol.y_events[0])
r_zl_tocka = np.array([event_f[:, 0], event_f[:, 1], event_f[:, 2]]).T
len_r_tocka =  np.linalg.norm(r_zl_tocka, axis=1)

alpha = np.arcsin(r_zl_tocka[:, 2].flatten() / len_r_tocka) * 180 / np.pi
orb_iter = np.ones(len(alpha)) * orb.cumsum()

def func(x, a, omega):
    return a * np.cos(omega * x)

optimizedParameters, pcov = opt.curve_fit(func, orb_iter, alpha, p0=[5.2, 0.001])
print(f"omega = {optimizedParameters[1]}")
print(f"precesija : {2 * np.pi / optimizedParameters[1] / 365.25} let")

plt.plot(orb_iter, alpha, label="Kot")
#plt.plot(orb_iter, np.arccos(alpha /  alpha.max()), c="g")
plt.plot(orb_iter, func(orb_iter, *optimizedParameters), 
         label=f"{round(optimizedParameters[0], 2)}° * cos({optimizedParameters[1]:.3e} *  1/(dan) * t)")
plt.ylabel("kot [°]")
plt.xlabel("t [dan]")
plt.title("kot(t)")
plt.legend()
plt.show()



ev_t = np.atleast_2d(np.array(sol.t_events[0]))
time = np.tile(sol.t, (len(ev_t), 1))
orb = (np.abs(time - ev_t.T).argmin(axis=1) + 1)[1:]
y = np.array_split(sol.y, orb, axis=1)[:-1]
t = np.array_split(sol.t, orb)[:-1]

indexes_kot = []
indexes_t = []

for v, time in zip(y, t):
    r = v[:3, :].T
    len_r = np.linalg.norm(r, axis = 1)
    index = len_r.argmax()
    tmp1 = r[index, :2]
    # kot = np.arccos(np.dot(tmp1, [0, 1])/ np.linalg.norm(tmp1))
    # if (tmp1[0] < 0): kot += np.pi
    tmp1 = tmp1 / np.linalg.norm(tmp1)
    #indexes_kot.append(kot * 180 / np.pi)
    indexes_kot.append(tmp1[0])
    indexes_t.append(time[index] / (60 * 60 * 24))

indexes_kot = np.array(indexes_kot)
indexes_t = np.array(indexes_t)

def func2(x, omega):
    return -1 * np.sin(omega * x)

optimizedParameters, pcov = opt.curve_fit(func2, indexes_t, indexes_kot, p0=[0.0019397512440925584])

print(f"omega2 : {optimizedParameters[0]}")
print(f"precesija2 : {2 * np.pi / optimizedParameters[0] / 365.25} let")

plt.plot(indexes_t, indexes_kot)
#plt.plot(indexes_t, func2(indexes_t, *optimizedParameters), label=f"-1 * sin({optimizedParameters[0]:.3e} * 1/(dan) * t)")
plt.ylabel("x [m]")
plt.xlabel("t[dnevi]")
plt.title("x(t)")
plt.legend(loc="upper left")
plt.show()