#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Module d'estimation par moindres carres
# David Coulot, octobre 2024
#

# Importation

import numpy as np

# Fonctions

# Estimation par moindres carres pour un modele explicite (Gauss-Markov)
    
def moindres_carres_GM(matA,vecB,flag,SigmaB=None):
    """
    Estimation par moindres carres pour un modele explicite
    
    En entree,
        - matA   = matrice modele (numpy array)
        - vecB   = vecteur de mesures (numpy array avec 2eme dimension)
        - flag   = booleen (vrai si SigmaB fournie)
        - SigmaB = matrice de variance-covariance des erreurs de mesure si connue (numpy array)
        
    En sortie,
        - Xchap    = vecteur des parametres estimes (numpy array avec 2eme dimension)
        - varXchap = matrice de variance-covariance des parametres estimes (numpy array)
        - Bchap    = vecteur des observations compensees (numpy array avec 2eme dimension)
        - Vchap    = vecteur des residus (numpy array avec 2eme dimension)
        - Vnor     = vecteur des residus normalises (numpy array avec 2eme dimension)
        - sigma02  = facteur unitaire de variance (flottant)
    """

    # Dimensions utiles
    
    n=matA.shape[0] # n = nombre d'observations
    p=matA.shape[1] # p = nombre de parametres
    
    # Matrice de poids
    
    if (flag):
        matP=np.linalg.inv(SigmaB) # P = SigmaB^(-1), sinon I_n
    
    # Systeme normal
    
    if (flag):
        matN=matA.T@matP@matA # N=A^T.P.A
        vecK=matA.T@matP@vecB # K=A^T.P.B
    else:
        matN=matA.T@matA
        vecK=matA.T@vecB
    
    # Solution des moindres carres
    
    matNinv=np.linalg.inv(matN) 
    Xchap=matNinv@vecK          # Xchap = N^(-1).K
    
    # Observations compensees et residus
    
    Bchap=matA@Xchap # Bchap = A.Xchap
    
    Vchap=vecB-Bchap # Vchap = B-Bchap = B-A.Xchap
    
    # Facteur unitaire de variance
    
    if (flag):
        sigma02=(Vchap.T@matP@Vchap)[0,0]/float(n-p) # sigma0^2 = Vchap^T.P.Vchap/(n-p)
    else:
        sigma02=(Vchap.T@Vchap)[0,0]/float(n-p)
    
    # Matrices de variance-covariance
    
    varXchap=sigma02*matNinv # var(Xchap) = sigma0^2.N^(-1)
    
    if (flag):
        varVchap=sigma02*(SigmaB-matA@matNinv@matA.T) # var(Vchap) = sigma0^2.(SigmaB-A.N^(-1).A^T)
    else:
        varVchap=sigma02*(np.eye(n)-matA@matNinv@matA.T)
    
    # Residus normalises
    
    Vnor=np.linalg.inv(np.sqrt(np.diag(np.diag(varVchap))))@Vchap # Vnor_i = Vchap_i/sqrt(varVchap_{i,i})
    
    # Renvoi des resultats
    
    return Xchap,varXchap,Bchap,Vchap,Vnor,sigma02

# Estimation par moindres carres pour un modele implicite (Gauss-Helmert)
    
def moindres_carres_GH(matA,matG,omega,vecB,flag,SigmaB=None):
    """
    Estimation par moindres carres pour un modele implicite
    
    En entree,
        - matA   = matrice des derivees partielles par rapport aux parametres (numpy array)
        - matG   = matrice des derivees partielles par rapport aux mesures (numpy array)
        - omega  = vecteur des ecarts de fermeture (numpy array avec 2eme dimension)
        - vecB   = vecteur de mesures (numpy array avec 2eme dimension)
        - flag   = booleen (vrai si SigmaB fournie)
        - SigmaB = matrice de variance-covariance des erreurs de mesure si connue (numpy array)
        
    En sortie,
        - Xchap    = vecteur des parametres estimes (numpy array avec 2eme dimension)
        - varXchap = matrice de variance-covariance des parametres estimes (numpy array)
        - Bchap    = vecteur des observations compensees (numpy array avec 2eme dimension)
        - Vchap    = vecteur des residus (numpy array avec 2eme dimension)
        - Vnor     = vecteur des residus normalises (numpy array avec 2eme dimension)
        - sigma02  = facteur unitaire de variance (flottant)
    """

    # Dimensions utiles
    
    r=omega.shape[0] # r = nombre de composantes de la fonction implicite
    p=matA.shape[1]  # p = nombre de parametres
    
    # Matrices et vecteurs utiles
    
    if (flag):
        matP=np.linalg.inv(matG@SigmaB@matG.T)
        matS=SigmaB@matG.T@matP
        SigmaBinv=np.linalg.inv(SigmaB)
    else:    
        matP=np.linalg.inv(matG@matG.T)
        matS=matG.T@matP
        
    matN=matA.T@matP@matA
    vecK=matA.T@matP@omega
    
    # Solution des moindres carres
    
    matNinv=np.linalg.inv(matN) 
    Xchap=-matNinv@vecK  
    
    # Residus et observations compensees
    
    if (flag):
        Vchap=SigmaB@matG.T@matP@(omega+matA@Xchap)
    else:
        Vchap=matG.T@matP@(omega+matA@Xchap)

    Bchap=vecB-Vchap
    
    # Facteur unitaire de variance
    
    if (flag):
        sigma02=(Vchap.T@SigmaBinv@Vchap)[0,0]/float(r-p)
    else:
        sigma02=(Vchap.T@Vchap)[0,0]/float(r-p)
    
    # Matrices de variance-covariance
    
    varXchap=sigma02*matNinv # var(Xchap) = sigma0^2.N^(-1)
    
    if (flag):
        varVchap=sigma02*(matS@matG@SigmaB-matS@matA@matNinv@matA.T@matS.T) 
    else:
        varVchap=sigma02*(matS@matG-matS@matA@matNinv@matA.T@matS.T) 
    
    # Residus normalises
    
    Vnor=np.linalg.inv(np.sqrt(np.diag(np.diag(varVchap))))@Vchap 
    
    # Renvoi des resultats
    
    return Xchap,varXchap,Bchap,Vchap,Vnor,sigma02

def LM(VecX, VecY, X0, f, df, prec=0.0000000001, P=None):
    A = df(X0, VecX)
    B = VecY - f(X0, VecX)
    
    if P is None : P = np.eye(len(VecY))
    
    phi_x0 = B.T @ P @ B
    
    N = A.T @ P @ A
    K = A.T @ P @ B
    
    lambd = 0.001
    N_augmented = N + lambd*np.diag(np.diag(N))
    
    dx = np.linalg.solve(N_augmented, K)
    
    phi = (VecY - f(X0 + dx, VecX)).T @ P @ (VecY - f(X0 + dx, VecX))
    
    PHI = [phi_x0[0, 0], phi[0, 0]]
    
    iteration = 1
    while np.abs(PHI[0] - PHI[1]) >= prec:
        iteration += 1
    
        if PHI[1] >= PHI[0] :
            print("ça empire...")
            lambd *= 10 
            
            N_augmented = N + lambd*np.diag(np.diag(N))
            dx = np.linalg.solve(N_augmented, K)
            
            phi = (VecY - f(X0 + dx, VecX)).T @ P @ (VecY - f(X0 + dx, VecX))
            
            PHI[1]=phi[0, 0]
        
        else:
            print("ça s'améliore")
            X0 += dx
            lambd /= 10
            
            A = df(X0, VecX)
            B = VecY - f(X0, VecX)
            
            N = A.T @ P @ A
            K = A.T @ P @ B
            
            N_augmented = N + lambd*np.diag(np.diag(N))
            dx = np.linalg.solve(N_augmented, K)
            
            phi = (VecY - f(X0 + dx, VecX)).T @ P @ (VecY - f(X0 + dx, VecX))
            
            phi_x0 = B.T @ P @ B
            
            PHI[0] = phi_x0
            PHI[1] = phi
            
            Xchap=X0
            
            n=len(VecY)
            p=len(Xchap)
            
            Vchap = B - A@Xchap
            
            sigma02 = Vchap.T@P@Vchap / (n-p)
            
            VarXchap = sigma02*np.linalg.inv(A.T@P@A)
            VarVchap = sigma02*(np.linalg.inv(P) - A@np.linalg.inv(A.T@P@A)@A.T)
            
            Vnor = np.linalg.inv(np.sqrt(np.diag(np.diag(VarVchap)))) @ Vchap

    return iteration, Xchap, Vchap, VarXchap, VarVchap, Vnor
    
    
    
    
    