import numpy as np
import matplotlib.pyplot as plt
import math
import scipy.integrate as spi

f = lambda x: 2.5 * np.cos(2 * x) * np.exp(2*x / 3) + 4*np.sin(3.5*x) * np.exp(-3*x) + 3*x

a = 0.1
b = 2.3
alpha = 1/5
beta = 0
eps = 1e-6

def S_lev(a, b, N):
    x = np.linspace(a, b, N+1)
    h = (b - a) / N
    return h * np.sum(f(x[:-1]))


def S_sred(a, b, N):
    h = (b - a) / N
    x_mid = np.linspace(a + h / 2, b - h / 2, N)
    return h * np.sum(f(x_mid))


def S_trap(a, b, N):
    X = np.linspace(a, b, N+1)
    h = (b - a)/N
    s = h*((f(a) + f(b))/2 + np.sum(f(X)[1:-1]))
    return s


def S_simp(a, b, N):
    X = np.linspace(a, b, N+1)
    x_mid = (X[:-1] + X[1:]) / 2
    h = (b - a)/N
    s = h/6 * np.sum(f(X)[:-1] + 4*f(x_mid) + f(X)[1:])
    return s

# Эйткен
m = lambda s1, s2, s3: -np.log(np.abs((s3 - s2)/(s2 - s1)))/np.log(2)



N_r = np.array([2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
R_lev = np.zeros(8)
R_sred = np.zeros(8)
R_trap = np.zeros(8)
R_simp = np.zeros(8)

for k in range(len(N_r)-2):
    N1 = N_r[k]; N2 = N_r[k+1]; N3 = N_r[k+2] # три сетки

    Ml = m(S_lev(a, b, N1),S_lev(a, b, N2),S_lev(a, b, N3))
    Mt = m(S_trap(a, b, N1),S_trap(a, b, N2),S_trap(a, b, N3))
    Ms = m(S_sred(a, b, N1),S_sred(a, b, N2),S_sred(a, b, N3))
    Msi = m(S_simp(a, b, N1),S_simp(a, b, N2),S_simp(a, b, N3))

    # 1ое приближение
    S11_lev = S_lev(a, b, N2) + (S_lev(a, b, N2) - S_lev(a, b, N1))/(2**Ml - 1)
    S12_lev = S_lev(a, b, N3) + (S_lev(a, b, N3) - S_lev(a, b, N2))/(2**Ml - 1)
    S11_sred = S_sred(a, b, N2) + (S_sred(a, b, N2) - S_sred(a, b, N1))/(2**Ms - 1)
    S12_sred = S_sred(a, b, N3) + (S_sred(a, b, N3) - S_sred(a, b, N2))/(2**Ms - 1)
    S11_trap = S_trap(a, b, N2) + (S_trap(a, b, N2) - S_trap(a, b, N1))/(2**Mt - 1)
    S12_trap = S_trap(a, b, N3) + (S_trap(a, b, N3) - S_trap(a, b, N2))/(2**Mt - 1)
    S11_simp = S_simp(a, b, N2) + (S_simp(a, b, N2) - S_simp(a, b, N1))/(2**Msi - 1)
    S12_simp = S_simp(a, b, N3) + (S_simp(a, b, N3) - S_simp(a, b, N2))/(2**Msi - 1)

    # 2ое приближение
    S2_lev = S12_lev + (S12_lev - S11_lev)/(2**(Ml+1) - 1)
    S2_sred = S12_sred + (S12_sred - S11_sred)/(2**(Ms+1) - 1)
    S2_trap = S12_trap + (S12_trap - S11_trap)/(2**(Mt+1) - 1)
    S2_simp = S12_simp + (S12_simp - S11_simp)/(2**(Msi+1) - 1)

    # погрешность
    R_lev[k] = (S2_lev - S12_lev)
    R_sred[k] = (S2_sred - S12_sred)
    R_trap[k] = (S2_trap - S12_trap)
    R_simp[k] = (S2_simp - S12_simp)

print('\n\n\n')
print('======================================================')
print(f'S левыми прямоугольниками = {S_lev(a, b, 1024)}')
print('------------------------------------------------------')
print(f'S средними прямоугольниками = {S_sred(a,b,1024)}')
print('------------------------------------------------------')
print(f'S трапециями = {S_trap(a,b,1024)}')
print('------------------------------------------------------')
print(f'S Симпсоном = {S_simp(a,b,1024)}')
print('======================================================')


# Абсолютная погрешность
error_lev = np.zeros(len(N_r))
error_sred = np.zeros(len(N_r))
error_trap = np.zeros(len(N_r))
error_simp = np.zeros(len(N_r))

fp = lambda x: (2.5 * np.cos(2 * x) * np.exp(2*x / 3) + 4*np.sin(3.5*x) * np.exp(-3*x) + 3*x) / (x - a)**alpha

ABS_error_fp = spi.quad(fp, a, b)[0]
ABS_error = spi.quad(f, a, b)[0]



for i in range(len(N_r)):
    error_lev[i] = np.abs(S_lev(a,b,N_r[i]) - ABS_error)
    error_sred[i] = np.abs(S_sred(a,b,N_r[i]) - ABS_error)
    error_trap[i] = np.abs(S_trap(a,b,N_r[i]) - ABS_error)
    error_simp[i] = np.abs(S_simp(a,b,N_r[i]) - ABS_error)

plt.figure(figsize=(9, 6))
plt.semilogy(N_r, error_lev,color = '#556B2F', label = 'Левые прямоугольники')
plt.semilogy(N_r, error_sred,color = '#000080', label = 'Средние прямоугольники')
plt.semilogy(N_r, error_trap,color = '#8B0000', label = 'Трапеции')
plt.semilogy(N_r, error_simp,color = '#7B68EE', label = 'Симпсон')
plt.grid(True,ls="--", alpha=0.5)
plt.legend()
plt.xlabel('Количество разбиений'); plt.ylabel('Абсолютная погрешность')
plt.savefig('abs_error_f4.png')
plt.show()

#=============================================================================
# 1.2
def MU(s, z1, z2):
    res = 0.0
    for k in range(s + 1):
        C = math.comb(s, k)
        term = (z2 - a)**(k + 1 - alpha) - (z1 - a)**(k + 1 - alpha)
        res += C * (a**(s - k)) * term / (k + 1 - alpha)
    return res

def Kardano(f,b_coeff,c,d):
    B = b_coeff / f; C = c / f; D = d / f
    q = 0.5 * (2*B**3 / 27 - B*C / 3 + D)
    p = (3*C - B**2) / 9
        
    r = np.sign(q) * np.sqrt(np.abs(p))
    if r == 0:
        r = 1e-15  # Защита от деления на ноль
            
    val = q / (r**3)
    val = np.clip(val, -1.0, 1.0) # Защита от вылета из [-1, 1] при округлении
    phi = np.arccos(val)
    y = np.zeros(3)
    y[0] = -2 * r * np.cos(phi / 3)
    y[1] =  2 * r * np.cos(np.pi / 3 - phi / 3)
    y[2] =  2 * r * np.cos(np.pi / 3 + phi / 3)
    x = y - B / 3
    return np.sort(x)

def NK(a,b,N):
    z = np.linspace(a,b,N+1); N_K = np.zeros(N)
    for i in range(N):
        mu = [MU(s,z[i],z[i+1]) for s in range(3)]
        x1 = z[i]; x2 = (z[i] + z[i+1])/2; x3 = z[i+1]
        M = np.array([mu[0],mu[1],mu[2]]); matr = np.array([
            [1,1,1],
            [x1,x2,x3],
            [x1**2,x2**2,x3**2]])
        A = np.linalg.solve(matr,M)
        N_K[i] = A[0]*f(x1) + A[1]*f(x2) + A[2]*f(x3)
    return np.sum(N_K)

def GS(a,b,N):
    z = np.linspace(a,b,N+1); Gauss = np.zeros(N)
    for i in range(N):
        mu = [MU(s,z[i],z[i+1]) for s in range(6)]
        Mu_m = np.array([[mu[0],mu[1],mu[2]],
                         [mu[1],mu[2],mu[3]],
                         [mu[2],mu[3],mu[4]]])
        a_G = np.linalg.solve(Mu_m, np.array([-mu[3],-mu[4],-mu[5]]))
        X = Kardano(1,a_G[2],a_G[1],a_G[0])
        g1 = X[0]; g2 = X[1]; g3 = X[2];
        M = np.array([mu[0],mu[1],mu[2]]);
        A_G = np.linalg.solve(np.array([[1,1,1],[g1,g2,g3],[g1**2,g2**2,g3**2]]),M)
        Gauss[i] = A_G[0]*f(g1) + A_G[1]*f(g2) + A_G[2]*f(g3)
    return np.sum(Gauss)


R_GS = np.zeros(len(N_r[:-2]) - 2)
R_NK = np.zeros(len(N_r[:-2]) - 2)
error_GS = np.zeros(len(N_r[:-2]))
error_NK = np.zeros(len(N_r[:-2]))

for k in range(len(N_r[:-2]) - 2):
    N1 = N_r[k]; N2 = N_r[k+1]; N3 = N_r[k+2] # три сетки
    
    # 1ое приближение
    GS1 = GS(a, b, N1); GS2 = GS(a, b, N2); GS3 = GS(a, b, N3)
    NK1 = NK(a, b, N1); NK2 = NK(a, b, N2); NK3 = NK(a, b, N3)

    error_GS[k] = np.abs(GS1 - ABS_error_fp); error_GS[k+1] = np.abs(GS2- ABS_error_fp); error_GS[k+2] = np.abs(GS3- ABS_error_fp);  
    error_NK[k] = np.abs(NK1- ABS_error_fp); error_NK[k+1] = np.abs(NK2- ABS_error_fp); error_NK[k+2] = np.abs(NK3- ABS_error_fp)

    Mgs = m(GS1, GS2, GS3)
    Mnk = m(NK1, NK2, NK3)
    S11_GS = GS2 + (GS2 - GS1)/(2**Mgs - 1)
    S12_GS = GS3 + (GS3 - GS2)/(2**Mgs - 1)
    S11_NK = NK2 + (NK2 - NK1)/(2**Mnk - 1)
    S12_NK = NK3 + (NK3 - NK2)/(2**Mnk - 1)

    # 2ое приближение
    S2_GS = S12_GS + (S12_GS - S11_GS)/(2**(Mgs+1) - 1)
    S2_NK = S12_NK + (S12_NK - S11_NK)/(2**(Mnk+1) - 1)

    # Погрешность
    R_GS[k] = np.abs(S2_GS - S12_GS)
    R_NK[k] = np.abs(S2_NK - S12_NK)


plt.figure(figsize=(9,6))
plt.semilogy(N_r[:-2],error_NK,'-*',color = '#800000', label = 'Ньютон-Котес')
plt.semilogy(N_r[:-2],error_GS,'-*',color = '#000080', label = 'Гаусс')
plt.xlabel('количество разбиений'); plt.ylabel('погрешность')
plt.legend()
plt.xlim(0, 128)
plt.grid(True, which="both", ls="--")
plt.savefig('abs_error_l2.png')
plt.show()


#7FFFD4
#=============================================================================
#2.1 and 2.2

R_GS2 = 10; R_NK2 = 10
power = 3
N1 = 0; N2 = 0; N3 = 0
N3_G = 0; N3_NK = 0
while R_GS2 > eps:
    power += 1
    N1 = 2**(power-2); N2 = 2**(power-1); N3 = 2**power
    GS1 = GS(a, b, N1); GS2 = GS(a, b, N2); GS3 = GS(a, b, N3)
    Mgs = m(GS1, GS2, GS3)

    # 1ое приближение
    S11_GS = GS2 + (GS2 - GS1)/(2**Mgs - 1)
    S12_GS = GS3 + (GS3 - GS2)/(2**Mgs - 1)

    # 2ое приближение
    S2_GS = S12_GS + (S12_GS - S11_GS)/(2**(Mgs+1) - 1)
    # Погрешность
    R_GS2 = np.abs(S2_GS - S12_GS)
    N3_G = N3

power2 = 3
N1 = 0; N2 = 0; N3 = 0
while R_NK2 > eps:
    power2 += 1
    N1 = 2**(power2-2); N2 = 2**(power2-1); N3 = 2**power2
    NK1 = NK(a, b, N1); NK2 = NK(a, b, N2); NK3 = NK(a, b, N3)
    Mnk = m(NK1, NK2, NK3)
    
    # 1ое приближение
    S11_NK = NK2 + (NK2 - NK1)/(2**Mnk - 1)
    S12_NK = NK3 + (NK3 - NK2)/(2**Mnk - 1)

    # 2ое приближение
    S2_NK = S12_NK + (S12_NK - S11_NK)/(2**(Mnk+1) - 1)

    # Погрешность
    R_NK2 = np.abs(S2_NK - S12_NK)
    N3_NK = N3


print('\n\n\n======================================================')
print('S = {GS(a,b,N3_G)}')
print('Погрешность по Ричардсону в методе Гаусса:',R_GS2)
print(f'h = {(b-a)/N3_G}')
print('------------------------------------------------------')
print(f'S = {NK(a,b,N3_NK)}')
print('Погрешность по Ричардсону в методе Ньютона-Котеса:',R_NK2)
print(f'h = {(b-a)/N3_NK}')
print('======================================================\n\n\n')
