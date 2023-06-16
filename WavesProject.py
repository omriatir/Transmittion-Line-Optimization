import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize
from scipy import optimize

#Question 1

def SolveTransmissionLine(N,Rg,l,L,C,f,ZL):
    w=2*np.pi*f
    dz = l/N
    factor_L = 1/(-2*dz*1j*w*L)
    factor_C = 1/(-2*dz*1j*w*C)
    M = np.zeros((2*N,2*N),dtype=complex)
    #Filling Egde conditions
    M[0][0]=1
    M[0][N]=Rg
    M[N-1][N-1]=1
    M[N-1][2*N-1]=-1*ZL
    M[2*N-1][N-1]=1/dz
    M[2*N-1][N-2]=-1/dz
    M[2*N-1][2*N-2]=(1j*w*L)/2
    M[2*N-1][2*N-1]=(1j*w*L)/2
    M[N][0]=-1/dz
    M[N][1]=1/dz
    M[N][N]=(1j*w*L)/2
    M[N][N+1]=(1j*w*L)/2
    #Fill the matrix
    for i in range(1,N-1):
            M[i][i]=1
            M[i][N+i-1]=factor_C
            M[i][N+i+1]=-factor_C
    for i in range(N+1,2*N-1):
            M[i][i]=1
            M[i][i-N-1]=factor_L
            M[i][i-N+1]=-factor_L
    b = np.zeros(2*N)
    b[0] = 1
    M_S = np.linalg.solve(M,b)
    V_array = M_S[0:N]
    I_array = M_S[N:2*N]
    Z_array = np.arange(0,N,1)*dz

    return V_array,I_array,Z_array

Solution = SolveTransmissionLine(2000,50,0.1,5*10**-7,8.88*10**-11,10**10,150)
N_a = Solution[2]

plt.plot(N_a,np.abs(Solution[0]))
plt.title("|V(z)| N=2000")
plt.xlabel("z [m]")
plt.ylabel("|V(z)| [V]")
plt.show()
plt.plot(N_a,np.abs(Solution[1]))
plt.xlabel("z [m]")
plt.ylabel("|I(z)| [A]")
plt.title("|I(z)| N=2000")
plt.show()
plt.plot(N_a,np.angle(Solution[0],deg=True))
plt.xlabel("z [m]")
plt.ylabel("phase(V(z)) [degree]")
plt.title("phase graph")
plt.show()
plt.plot(N_a,np.angle(Solution[1],deg=True))
plt.xlabel("z [m]")
plt.ylabel("phase(I(z)) [degree]")
plt.title("phase graph")
plt.show()

#Question 2
l=0.1
L_0 = 5*10**-7
C_0 = 8.88*10**-11
C_1 = 4.7140452*10**-12
L_1 = 5.3033008*10**-8
R_L = 150
Z_0 = 75
c = 3*10**8
f = 10**10
beta = 2*f*np.pi*np.sqrt(L_1*C_1)
Z_1 = np.sqrt(L_1/C_1)


def Gamma_In_abs(f):
    return abs((Z_1*(R_L+Z_1*1j*np.tan(2*np.pi*f*(L_1*C_1)**0.5*l/2))/(Z_1+1j*R_L*np.tan(2*np.pi*f*(L_1*C_1)**0.5*l/2))-Z_0)
               /(Z_1*(R_L+Z_1*1j*np.tan(2*np.pi*f*(L_1*C_1)**0.5*l/2))/(Z_1+1j*R_L*np.tan(2*np.pi*f*(L_1*C_1)**0.5*l/2))+Z_0))

f_array = np.linspace(0.5*10**10,1.5*10**10,1000)
ref = np.ones(len(f_array))*0.1
Gamma_Array = Gamma_In_abs(f_array)

#plot analytical solution - Descrete Rollback Towards Generator.

plt.plot(f_array,Gamma_Array,f_array,ref)
plt.title("|Gamma_In|")
plt.xlabel("frequency [Hz]")
plt.ylabel("|Gamma_In|")
plt.show()

def L(z,alpha_1,L1):
    return L1+alpha_1*((z-l/2)**2)

def C(z,alpha_2,C1):
    return C1+alpha_2*((z-l/2)**2)

# Try to Get Z_IN to the differential Line:
w = 10**10
N=200
dz=l/N

def get_GAMMA_IN_2(alpha1,alpha2,C1,L1,w_array):
    Z_temp = R_L
    for i in range(N,N//2,-1):
        L_temp = L(i*dz,alpha1,L1)
        C_temp = C(i*dz,alpha2,C1)
        beta = 2*np.pi*w_array*np.sqrt(L_temp*C_temp)
        Zc = np.sqrt(L_temp/C_temp)
        Z_in = Zc*(((Z_temp)+1j*Zc*np.tan(beta*dz))/((Zc+1j*Z_temp*np.tan(beta*dz))))
        Z_temp = Z_in
    GAMMA_IN = np.abs((Z_temp - Z_0)/(Z_temp + Z_0))
    return GAMMA_IN

# plot Gamma_In(f)

f_array_2 = np.linspace(0.02*10**10,3*10**10,5000)
ref_2 = np.ones(len(f_array_2))*0.1

# find MAX AND MIN Frequency And Bandwidth
def BandWidth(params):
    f_under = f_array_2[f_array_2 < 10**10]
    f_above = f_array_2[f_array_2 > 10**10]
    Gamma_Array_under = get_GAMMA_IN_2(params[0], params[1], params[2], params[3], f_under)
    Gamma_Array_above = get_GAMMA_IN_2(params[0], params[1], params[2], params[3], f_above)
    left_intersections = f_under[np.argwhere(np.diff(np.sign(0.1 - Gamma_Array_under))).flatten()]
    right_intersections = (f_above[np.argwhere(np.diff(np.sign(0.1 - Gamma_Array_above))).flatten()])
    if get_GAMMA_IN_2(params[0],params[1],params[2],params[3],10**10)>0.1:
        return 0
    if len(right_intersections)!=0 and len(left_intersections)!=0 :
        RBW = np.abs(10 * 10 - min(right_intersections))
        LBW = np.abs(10 ** 10 - max(left_intersections))
        return -2*min(RBW,LBW)
    return 0

# Finding ALPHA Parameters and L1,C1 that Minimize BANDWIDTH (UNCOMMENT if you want to search initial guesses):
# ar = []
# for i in range(10,15):
#     for j in range(13,19):
#         for k in range(1,9):
#             for p in range(1,9):
#                 minimized_params = optimize.minimize(BandWidth,np.array([(2+0.1*i)*10**-5,(2+0.1*j)*10**-10,C_1*k/2,L_1*p/2]),method='Nelder-Mead',
#                                      bounds = [(0,1),(0,1),(0,1),(0,1)])
#                 ar.append(int(str(i)+str(j)+str(k)+str(p)))
#                 ar.append(minimized_params.fun)
#
# np_array_view = np.array(ar)
# print(ar[(np.argmin(np.array(ar))-1)])
# print(ar)
# print(ar[np.argmin(np_array_view)])

results_final = optimize.minimize(BandWidth,np.array([(2+0.1*13)*10**-5,(2+0.1*16)*10**-10,C_1*4/2,L_1*2/2]),method='Nelder-Mead',
                                     bounds = [(0,1000),(0,1000),(0,1),(0,1)])
print(results_final)
alpha_1_final = results_final.x[0]
alpha_2_final = results_final.x[1]
L_1_best_final = results_final.x[3]
C_1_best_final = results_final.x[2]

alpha_1_array_final = np.ones(len(f_array_2)) * alpha_1_final
alpha_2_array_final = np.ones(len(f_array_2)) * alpha_2_final
L_1_best_array_final = np.ones(len(f_array_2)) * L_1_best_final
C_1_best_array_final = np.ones(len(f_array_2)) * C_1_best_final

Gamma_Array_2 = get_GAMMA_IN_2(alpha_1_array_final,alpha_2_array_final,C_1_best_array_final,L_1_best_array_final,f_array_2)

plt.plot(f_array_2,Gamma_Array_2,f_array_2,ref_2,f_array,Gamma_Array)
plt.title("|Gamma_In|")
plt.xlabel("frequency [Hz]")
plt.legend(["maximized parameters","reference","Alpha 1,2 = 0  L1,C1 from a section"])
plt.ylabel("|Gamma_In|")
plt.show()































