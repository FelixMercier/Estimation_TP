import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

import Module_Estimation as me

####       EX 2
data = np.genfromtxt("Exercice_2_Data.dat", delimiter = " ")
VecX, VecY = data[:,0].reshape((len(data), 1)), data[:,1].reshape((len(data), 1))

def f(X0, x):
    a0, b0 = X0[0], X0[1]
    return 1/(1 + a0* np.exp(-b0*x))

def f_lin(X0, X):
    return X0[0] * X + X0[1]

def dfda(X0, x):
    a0, b0 = X0[0], X0[1]
    num = -np.exp(-b0*x)
    denom = (1 + a0*np.exp(-b0*x))**2
    return num/denom

def dfdb(X0, x):
    a0, b0 = X0[0], X0[1]
    num = a0*x*np.exp(-b0*x)
    denom = (1 + a0*np.exp(-b0*x))**2
    return num / denom

def df(X0, x):
    """return the jacobian matrice of f"""
    return np.hstack((dfda(X0, x), dfdb(X0, x)))

if __name__ == "__main__":
    
    question = "3"
    
    if question == "1":
        idx = np.argwhere(VecY < 1)
        VecY = VecY[idx[:, 0]]
        VecX = VecX[idx[:, 0]]
        
        VecYbis = np.log((1/VecY) - 1)
        
        P = np.log((1/0.05) - 1) * np.eye(len(VecX))
        
        A = np.hstack((-VecX, np.ones((len(VecX), 1))))
        B = VecYbis
        
        Xchap,varXchap,Bchap,Vchap,Vnor,sigma02 = me.moindres_carres_GM(A, B, flag=False)
        
        
        plt.scatter(VecX, VecY, c='r')
        reY = 1 / (1 + np.exp(Xchap[0]*(-VecX) + Xchap[1]))
        plt.plot(VecX, reY)
        plt.title("Estimation par moindre carré classique après linéarisation : a = {} et b = {}".format(-Xchap[0], Xchap[1]))
        plt.show()        
    
    if question == "2" :
        a0, b0 = 1, 0.5
        X0 = np.array([[a0], [b0]])        
        A = np.hstack((dfda(X0, VecX), dfdb(X0, VecX)))
        
        B = VecY - f(X0, VecX)
        
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
            A = np.hstack((dfda(X0, VecX), dfdb(X0, VecX)))
            B =  VecY - f(X0, VecX)
            
        
            
        print("Résultats de l'estimation: ")
        print(" ")
        print("en {} itérations".format(iterations))
        print(" ")
        print("a = {} et b = {}".format(np.round(a0, 5), np.round(b0, 5)))
        print("fracteur unitaire de variance = {}".format(np.round(sigma02, 4)))
        
            
        fig3 = plt.figure()
        plt.scatter(VecX, VecY, c="r", s=10)
        plt.plot(VecX, f(X0, VecX))
        plt.title("après {} itérations".format(iterations))
        plt.show()
        
    if question == "3":
        print("question 3 :")
        print(" ")
        
        a0, b0 = 1, 0.5
        X0 = np.array([[a0], [b0]])
        iteration, Xchap, Vchap, VarXchap, VarVchap, Vnor = me.LM(VecX, VecY, X0, f, df)
        print("fait")
        print(iteration, X0)
        
        fig = plt.figure()
        plt.scatter(VecX, VecY, c='r', s=10)
        plt.plot(VecX, f(Xchap, VecX), c='skyblue')
        plt.title("Estimatiuon par l'algorithme de Levenberg Marquadt, {} itérations, a = {}, b = {}".format(iteration, Xchap[0], Xchap[1]))
        
        fig_res = plt.figure()
        plt.scatter(VecX, Vchap)
        plt.title("Visualisation des résidus")
        
        fig_hist = plt.figure()
        plt.hist(Vnor, bins=15, density=True, edgecolor='black', color='b', alpha=0.6)
        lin = np.linspace(-4, 4, 2000)
        mean, std = norm.fit(Vnor)
        plt.plot(lin, norm.pdf(lin, mean, std), c='black')
        
        plt.show()
        
        