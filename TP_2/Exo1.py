import numpy as np
import matplotlib.pyplot as plt
import Fonctions_TP as tp
from scipy.optimize import linprog
from scipy.stats import norm

######

def simplex(A, nb_para, B):
    """
    A : matrice A du problème B = AX + V
    X : idem
    B : idem, les b_i doivent être positifs ou nuls
    """
    
    # V = B - A @ X

    # U = [V[i] if V[i] >= 0 else 0 for i in range(0, len(V))]
    # W = [-V[i] if V[i]<=0 else 0 for i in range(0, len(V))]
    
    # X1 = [X[i] if X[i] >= 0 else 0 for i in range(0, len(X))]
    # X2 = [-X[i] if X[i]<=0 else 0 for i in range(0, len(X))]
    
    c = np.hstack(( np.zeros((2*nb_para,)) , np.ones((2*len(B),)) ))
    
    A_eq = np.hstack((A, -A, np.eye(len(A)), np.eye(len(A))))
    B_eq = B
    
    sol = linprog(c, A_eq=A_eq, b_eq=B_eq, method='simplex')
    
    n_iter = sol.nit
    para = sol.x
    Xchap = para[0:nb_para] - para[nb_para : 2* nb_para]
    Vchap = B - A@Xchap
    
    return n_iter, Xchap, Vchap

def ransac(A, B, P, n, t, T, K):
    nb_bonnes_mes_abs = [1]
    choix_mes_abs = np.array([[0], [0]])
    
    for i in range(K):
        
        choixmes = np.random.choice(100, n, replace = False)
        
        Xchap, Vchap = tp.moindres_carres(A, B, P, choixmes)
        
        # St = [Vchap[j] if np.abs(Vchap[j]) < t else 0 for j in range(len(Vchap))]
        St = [j for j in range(len(Vchap)) if np.abs(Vchap[j]) < t]
        
        nb_bonnes_mes = len(St)
        if nb_bonnes_mes >= T :
            Xchapf, Vchapf = tp.moindres_carres(A, B, P, St)
            return Xchapf, Vchapf, i
        
        elif nb_bonnes_mes > nb_bonnes_mes_abs[0]: 
            nb_bonnes_mes_abs[0] = nb_bonnes_mes
            choix_mes_abs = St
            
    Xchap, Vchap = tp.moindres_carres(A, B, P, choix_mes_abs)
    
    return Xchap, Vchap, K


xmes, ybruit = tp.creer_mesures(2, 5, -50, 50, 100, 1, 0, 0, 0)
vecx = xmes.reshape(len(xmes), 1)
vecy = ybruit.reshape(len(ybruit), 1)

A = np.hstack(( vecx, np.ones(np.shape(vecx)) ))

nb_para = 2

B = np.abs(vecy)


sigma = 5
P = sigma*np.eye(len(vecx))



        
if __name__ == '__main__' : 
    
    estimation_method = 'simplex'
    
    if estimation_method == 'ransac':
        
        print('Estimation par ransac')
        
        Xchap, Vchap, n_iter = ransac(A, vecy, P, 2, 1.96, 95, 100)

        esti_ransac = plt.figure()
        
        plt.scatter(vecx, vecy, c='red', label='mesures')
        plt.plot(vecx, Xchap[0] * vecx + Xchap[1], c='blue', label='modèle')
        
        plt.title("Resultat de l'algo ransac, a = {} et b = {}, après {} iterations".format(np.round(Xchap[0], 4), np.round(Xchap[1], 4), n_iter))
                
        resi_ransac = plt.figure()
        plt.hist(Vchap, density=True, bins = 20, color='skyblue', edgecolor='black')
        lin = np.linspace(-4, 4, 1500)
        mean = np.mean(Vchap)
        std = np.std(Vchap)
        plt.plot(lin, norm.pdf(lin, mean, std))
        plt.title("Résidus de l'algo Ransac")
        
    if estimation_method == 'simplex':
        
        print("Estimation par simplex modifié")
        
        n_iter, Xchap, Vchap = simplex(A, nb_para, B)

        est =  plt.figure()
        
        plt.scatter(xmes, ybruit, c='red')
        plt.plot(vecx, Xchap[0]* vecx + Xchap[1])
        
        plt.title('Estimation para simplex modifié, {} itérations'.format(n_iter))
        
        # res = plt.figure()
        # plt.scatter(vecx, Vchap)
        # plt.title('Résidus')
        
        print("a = {}".format(np.round(Xchap[0], 4)))
        print("b = {}".format(np.round(Xchap[1], 4)))
        
        plt.show()
        
                
        
        
        
        
        
        