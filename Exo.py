import numpy as np
import matplotlib.pyplot as plt

import Module_Estimation as me

##########     EX 1 

#importing data

data = np.genfromtxt("Exercice_1_Data_1.dat", delimiter=" ")
VecX, VecY = data[:,0].reshape((len(data), 1)), data[:,1].reshape((len(data), 1)) 

#Gauss Markov
A_GM = data[:, 0]

X_GM = data[:, 0]

A_GM = np.hstack((np.reshape(A_GM, (len(A_GM), 1)), np.ones((len(A_GM), 1))))

B_GM = data[:, 1].reshape((len(data), 1))

sigmaB_GM = 25 * np.eye(len(B_GM))

Xchap_GM,varXchap_GM,Bchap_GM,Vchap_GM,Vnor_GM,sigma02_GM = me.moindres_carres_GM(A_GM, B_GM, True, sigmaB_GM)
a0_GM, b0_GM = Xchap_GM[0,0], Xchap_GM[1,0]

fig1 = plt.figure()

# plt.scatter(VecX, VecY, c="red", s=10)
# plt.plot(VecX, a0_GM*VecX + b0_GM)
# plt.title("Estimation explicite")
# plt.show()

# fig11 = plt.figure()
# plt.hist(Vnor_GM, bins=20, color="skyblue", edgecolor="black")
# plt.title("Résidus normalisés, Esimation explicite")





#Gauss Helmert
a0 = 2
b0 = 5

A_GH = A_GM

B_GH = np.vstack((VecX, VecY))

G_GH = np.hstack((a0*np.eye(len(data)), (-1)*np.eye(len(data))))

x, y = data[:,0].reshape((len(data), 1)), data[:, 1].reshape((len(data), 1))
omega_GH = a0*x + (b0-y)

sigmaB_GH = 25*np.eye(2*len(data))

Xchap_GH = np.array([[1], [1]])
i = 0
while np.abs(Xchap_GH[0, 0]) > 0.000001 and np.abs(Xchap_GH[1, 0]) > 0.000001:
    i+=1
    Xchap_GH,varXchap_GH,Bchap_GH,Vchap_GH,Vnor_GH,sigma02_GH = me.moindres_carres_GH(A_GH, G_GH, omega_GH, B_GH , True, sigmaB_GH)
    a0 = Xchap_GH[0, 0] + a0
    b0 = Xchap_GH[1, 0] + b0
    G_GH =  np.hstack((a0*np.eye(len(data)), (-1)*np.eye(len(data))))
    omega_GH = a0*x + (b0-y)
    
a0_GH, b0_GH = a0, b0

fig2 = plt.figure()

# plt.scatter(VecX, VecY, c="red", s=10)
# plt.plot(VecX, a0_GM*VecX + b0_GM)
# plt.title("Estimation implicite")
# plt.show()

# fig21 = plt.figure()
# plt.hist(Vnor_GH, bins=20, color="skyblue", edgecolor="black")
# plt.title("Résidus normalisés, Esimation implicite")


####       EX 2
data2 = np.genfromtxt("Exercice_2_Data.dat", delimiter = " ")
VecX, VecY = data2[:,0].reshape((len(data2), 1)), data2[:,1].reshape((len(data2), 1))

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

P = np.eye(len(data2))

iterations = 0
sigma02 = 1
s02 = [40, 50] #valeur de sgma02 à litération n-1 puis à l'itération n
while np.abs(s02[0] - s02[1]) > 0.0000000001:
    iterations += 1
    Xchap,varXchap,Bchap,Vchap,Vnor,sigma02 = me.moindres_carres_GM(A, B, False)
    print(Xchap)
    s02[0] = s02[1]
    s02[1] = sigma02
    a0 = a0 + Xchap[0,0]
    b0 = b0 + Xchap[1, 0]
    A = np.hstack((dfda(a0, b0, VecX), dfdb(a0, b0, VecX)))
    B =  VecY - f(a0, b0, VecX)
    
fig3 = plt.figure()
plt.scatter(VecX, VecY, c="r", s=10)
plt.plot(VecX, f(a0, b0, VecX))
plt.title("après {} itérations".format(iterations))
    
    


    
    