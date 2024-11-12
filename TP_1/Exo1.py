import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

import Module_Estimation as me

##########     EX 1 

#importing data

# data = np.genfromtxt("Exercice_1_Data_1.dat", delimiter=" ")
data = np.genfromtxt("Exercice_1_Data_2.dat", delimiter=" ")
VecX, VecY = data[:,0].reshape((len(data), 1)), data[:,1].reshape((len(data), 1)) 

#Gauss Markov


A_GM = np.hstack((VecX, np.ones((len(VecX), 1))))

B_GM = VecY

sigmaB_GM = 25 * np.eye(len(B_GM))

Xchap_GM,varXchap_GM,Bchap_GM,Vchap_GM,Vnor_GM,sigma02_GM = me.moindres_carres_GM(A_GM, B_GM, True, sigmaB_GM)
a0_GM, b0_GM = Xchap_GM[0,0], Xchap_GM[1,0]

fig1 = plt.figure()

plt.scatter(VecX, VecY, c="red", s=10)
plt.plot(VecX, a0_GM*VecX + b0_GM)
plt.title("Estimation explicite, a = {} et b = {}".format(np.round(a0_GM, 3), np.round(b0_GM, 3)))

arg = np.argsort(VecX.reshape((1,201)))
VecXsorted = VecX[arg][0]
VecYsorted = VecY[arg][0]

fig14 = plt.figure()
plt.scatter(VecX, VecY, s=10)
plt.plot(VecXsorted, VecYsorted, c="red")
plt.title("Observations")

fig11 = plt.figure()
plt.hist(Vnor_GM, bins=10, density=True, color="skyblue", edgecolor="black")
lin = np.linspace(-4, 4, 1500)
mean = np.mean(Vnor_GM)
std = np.std(Vnor_GM)
plt.plot(lin, norm.pdf(lin, mean, std))
plt.title("Résidus normalisés, Esimation explicite")


fig12 = plt.figure()
plt.scatter(VecX, Vchap_GM, c='red')
plt.title("Résidus")


print("Résultats de l'estimation EXPLICITE : ")
print(" ")
print("a = {} et b = {}".format(np.round(a0_GM, 5), np.round(b0_GM, 5)))
print("fracteur unitaire de variance = {}".format(np.round(sigma02_GM, 4)))


#Gauss Helmert
a0 = 1.5
b0 = 4.0

A_GH = A_GM

B_GH = np.vstack((VecX, VecY))

G_GH = np.hstack((a0*np.eye(len(data)), (-1)*np.eye(len(data))))

x, y = data[:,0].reshape((len(data), 1)), data[:, 1].reshape((len(data), 1))
omega_GH = a0*x + (b0-y)

sigmaB_GH = np.eye(2*len(data))
sigmaB_GH[0:201] = sigmaB_GH[0:201]*9
sigmaB_GH[201:] = sigmaB_GH[201:]*25


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

plt.scatter(VecX, VecY, c="red", s=10)
plt.plot(VecX, a0_GM*VecX + b0_GM)
plt.title("Estimation implicite")

fig21 = plt.figure()
plt.hist(Vnor_GH, bins=10, density=True, color="skyblue", edgecolor="black")
lin = np.linspace(-4, 4, 1500)
mean = np.mean(Vnor_GH)
std = np.std(Vnor_GH)
plt.plot(lin, norm.pdf(lin, mean, std))
plt.title("Résidus normalisés, Esimation implicite")

fig22 = plt.figure()
plt.scatter(np.arange(0, 402), Vchap_GH)
plt.title("Résidus méthode implicite")

plt.show()



print(" ")
print(" ")
print("Résultats de l'estimation IMPLICITE: ")
print(" ")
print("en {} itérations".format(i))
print(" ")
print("a = {} et b = {}".format(np.round(a0_GH, 5), np.round(b0_GH, 5)))
print("fracteur unitaire de variance = {}".format(np.round(sigma02_GH, 4)))


    
    


    
    