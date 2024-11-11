import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

import Module_Estimation as me

####       EX 2
data = np.genfromtxt("Exercice_2_Data.dat", delimiter = " ")
VecX, VecY = data[:,0].reshape((len(data), 1)), data[:,1].reshape((len(data), 1))

def f(a0, b0, x):
    return 1/(1 + a0* np.exp(-b0*x))

def dfda(a0, b0, x):
    num = -np.exp(-b0*x)
    denom = (1 + a0*np.exp(-b0*x))**2
    return num/denom

def dfdb(a0, b0, x):
    num = a0*x*np.exp(-b0*x)
    denom = (1 + a0*np.exp(-b0*x))**2
    return num / denom

a0, b0 = 1, 0.5

A = np.hstack((dfda(a0, b0, VecX), dfdb(a0, b0, VecX)))

B = VecY - f(a0, b0, VecX)

P = np.eye(len(data))

iterations = 0
sigma02 = 1
s02 = [40, 50] #valeur de sgma02 à litération n-1 puis à l'itération n
while np.abs(s02[0] - s02[1]) > 0.0000000001:
    iterations += 1
    Xchap,varXchap,Bchap,Vchap,Vnor,sigma02 = me.moindres_carres_GM(A, B, True, P)
    print(Xchap)
    s02[0] = s02[1]
    s02[1] = sigma02
    a0 = a0 + Xchap[0,0]
    b0 = b0 + Xchap[1, 0]
    A = np.hstack((dfda(a0, b0, VecX), dfdb(a0, b0, VecX)))
    B =  VecY - f(a0, b0, VecX)
    

    
print("Résultats de l'estimation: ")
print(" ")
print("en {} itérations".format(iterations))
print(" ")
print("a = {} et b = {}".format(np.round(a0, 5), np.round(b0, 5)))
print("fracteur unitaire de variance = {}".format(np.round(sigma02, 4)))

    
fig3 = plt.figure()
plt.scatter(VecX, VecY, c="r", s=10)
plt.plot(VecX, f(a0, b0, VecX))
plt.title("après {} itérations".format(iterations))
plt.show()