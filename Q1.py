import numpy as np
import matplotlib.pyplot as plt
import sympy
import pandas as pd

# consider number in Michigan
# assume M = 0.7*N, F=0.3*N
# F[0] means number of female in 1061
# dF[0] means number of 1962-1961
# dF[-1] means number of 1961-1960
N_prey = [392100,  # 1961
          775161,  # 1962
          1185755,  # 1963
          1195770,  # 1964
          826900]  # 1965

M = []
F = []
dN_prey = []
dM = []
dF = []
N_pred = []
capture_N_pred = [35980,
                  5610,
                  7267,
                  4772,
                  3922,  # modify
                  3580,
                  2779,
                  5109]
for i in range(len(capture_N_pred)):
    N_pred.append(capture_N_pred[i] * 2)

for i in range(0, 8):
    M.append(N_pred[i] * 0.7)
    F.append(N_pred[i] * 0.3)

for i in range(0, 4):
    dN_prey.append(N_prey[i + 1] - N_prey[i])
    dF.append(F[i + 1] - F[i])
    dM.append(M[i + 1] - M[i])

# dF.append((F[0] - F[-1]))
# dM.append((M[0] - M[-1]))
K_R = 1508488
K_F = 468700000000000000000  # 环境承载力

# equations about N


r_pery, alpha = sympy.symbols("x,y")
eq1 = N_prey[2] * r_pery * (1 - N_prey[2] / K_R) - N_prey[2] * N_pred[2] * alpha - dN_prey[1]
eq2 = N_prey[3] * r_pery * (1 - N_prey[3] / K_R) - N_prey[3] * N_pred[3] * alpha - dN_prey[2]
s = sympy.solve([eq1, eq2], [r_pery, alpha])
print(list(s.values()))

# equations about F


r_female, beta_female, k_male = sympy.symbols("a,b,c")
# eq3_index = 4
eq3 = r_female * F[4] * (1 - F[4] / K_F) + beta_female * F[4] * N_prey[4] + k_male * M[4] - dF[3]
eq4 = r_female * F[3] * (1 - F[3] / K_F) + beta_female * F[3] * N_prey[3] + k_male * M[3] - dF[2]
eq5 = r_female * F[2] * (1 - F[2] / K_F) + beta_female * F[2] * N_prey[2] + k_male * M[2] - dF[1]
t = sympy.solve([eq3, eq4, eq5], [r_female, beta_female, k_male])
print(t)

# equations about M
# K_M = 4687 * 2 * 0.7  # 环境承载力
#
# r_male, beta_male, k_female = sympy.symbols("a,b,c")
# eq6 = r_male * M[4] * (1 - M[4] / K_M) + beta_male * M[4] * N_prey[4] + k_female * F[4] - dM[3]
# eq7 = r_male * M[3] * (1 - M[3] / K_M) + beta_male * M[3] * N_prey[3] + k_female * F[3] - dM[2]
# eq8 = r_male * M[2] * (1 - M[2] / K_M) + beta_male * M[2] * N_prey[2] + k_female * F[2] - dM[1]
# t = sympy.solve([eq6, eq7, eq8], [r_male, beta_male, k_female])
# print(t)

# r_male = r_female = r
# beta_male = beta_female = beta
# r, beta, k_F, k_M = sympy.symbols("x,y,z,w")
# eq9 = r * F[1] + beta * F[1] * N_prey[1] + k_F * M[1] - dF[0]
# eq10 = r * F[2] + beta * F[2] * N_prey[2] + k_F * M[2] - dF[1]
# eq11 = r * M[1] + beta * M[1] * N_prey[1] + k_M * F[1] - dM[0]
# eq12 = r * M[2] + beta * M[2] * N_prey[2] + k_M * F[2] - dM[1]
#
# eq13 = r * F[0] + beta * F[0] * N_prey[0] + k_F * M[0] - dF[-1]
# eq14 = r * M[0] + beta * M[0] * N_prey[0] + k_M * F[0] - dM[-1]
# print(F)
# print(M)
# print(eq9)
# print(eq10)
# print(eq11)
# print(eq12)
# print(eq13)
# print(eq14)
# t = sympy.solve([eq9, eq10, eq13], [r, beta, k_F, k_M])
# print(t)
# exit(0)
# argument
r_prey = 0.964567701518982
alpha = 6.82544520966953e-6
r = -2.0213353573062
beta = 1.50939093249243e-6
k_M = 0.538704624743195


# k_F = 7/3*7/3*k_M


# 第0年为1962年

def funcNt(n, f):
    return r_prey * n * (1 - n / K_R) - alpha * n * (f * 10 / 3)


def funcFt(n, f):
    # print(r * f * (1 - f / K_F) + beta * f * n + k_M * (f * 7 / 3))
    return r * f * (1 - f / K_F) + beta * f * n + k_M * (f * 7 / 3)


print("equation")
print(funcFt(N_prey[2], N_pred[2] * 0.3))
print(dF[1])

t_min = 0  # start at year
t_max = 25  # end at year 10 (1971)
t_h = 1e-1

t = np.linspace(t_min, t_max, int((t_max - t_min) / t_h + 1))
n = t.copy()
f = t.copy()
n[0] = N_prey[1]
f[0] = F[1]
# h = t_h
for i in range(t.shape[0] - 1):
    k1_n = funcNt(n[i], f[i])
    k1_f = funcFt(n[i], f[i])

    k2_n = funcNt(n[i] + k1_n * t_h / 2.0, f[i])
    k2_f = funcFt(n[i], f[i] + k1_f * t_h / 2.0)

    k3_n = funcNt(n[i] + k2_n * t_h / 2.0, f[i])
    k3_f = funcFt(n[i], f[i] + k2_f * t_h / 2.0)

    k4_n = funcNt(n[i] + k3_n * t_h, f[i])
    k4_f = funcFt(n[i], f[i] + k3_f * t_h)

    n[i + 1] = n[i] + t_h / 6.0 * (k1_n + 2.0 * k2_n + 2.0 * k3_n + k4_n)
    f[i + 1] = f[i] + t_h / 6.0 * (k1_f + 2.0 * k2_f + 2.0 * k3_f + k4_f)
    # print(k1_f,k2_f,k3_f,k4_f)

print(f[0] * 10 / 3)
print(f[int(t.shape[0] / 3)] * 10 / 3)
print(f[int(t.shape[0] / 3 * 2)] * 10 / 3)
print(f[int(t.shape[0] - 1)] * 10 / 3)
print("n")
print(n[0])
print(n[int(t.shape[0] / 3)])
print(n[int(t.shape[0] / 3 * 2)])
print(n[int(t.shape[0] - 1)])

# 2008488 11538
print("delta N")
print(funcNt(2008488, 11538))
print("delta F")
print(funcFt(2008488, 11538))

data = {
    't':t,
    'N_prey':n,
    'F':f
}
df = pd.DataFrame(data)
df.to_excel('output.xlsx', index=False)


plt.subplot(1, 2, 1)
plt.plot(t, n, 'b', label='N_prey')
plt.legend()
plt.xlabel('t_year')
plt.ylabel('number')
plt.title('n-t')
# t[int(t.shape[0]/8*4):int(t.shape[0]/8*8)], n[int(t.shape[0]/8*4):int(t.shape[0]/8*8)]
plt.subplot(1, 2, 2)
plt.plot(t, f, 'r--',
         label='F')
plt.legend()
plt.xlabel('t_year')
plt.ylabel('number')
plt.title('n-t')
plt.show()
