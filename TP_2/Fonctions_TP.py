#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# D. Coulot
# Fonctions utiles pour le TP du 12/11/2024

# Importations

import numpy as np

# Fonctions

# Fonction pour generer un jeu de donnees bruitees
# En entree,
#   - val_a     = valeur vraie de la pente de la droite
#   - val_b     = valeur vraie de l'ordonnee a l'origine de la droite
#   - x_min     = valeur minimale des abscisses
#   - x_max     = valeur maximale des abscisses
#   - nbr_mes   = nombre de mesures a creer
#   - sigma_mes = ecart-type du bruit de mesure centre gaussien
#   - prop_err  = pourcentage d'erreurs a introduire dans les mesures
#   - mu_err    = moyenne des erreurs a introduire dans les mesures
#   - sigma_err = ecart-type des erreurs a introduire dans les mesures
#
# En sortie,
#   - x_mes     = temps de mesures
#   - y_bruit   = mesures bruitees

def creer_mesures(val_a,val_b,x_min,x_max,nbr_mes,sigma_mes,prop_err,mu_err,sigma_err):
    
    # Mesures vraies
    
    x_mes=np.linspace(x_min,x_max,nbr_mes)
    y_vrai=val_a*x_mes+val_b
    
    # Bruit de mesure

    if (sigma_mes==0.):
        bruit=np.zeros(nbr_mes)
    else:
        bruit=sigma_mes*np.random.randn(nbr_mes)
        
    # Erreurs de mesure
    
    if (prop_err>0.):
        
        nbr_err=int(float(nbr_mes)*prop_err/100.)
        
        indices=np.random.choice(nbr_mes,nbr_err,replace=False)
        
        bruit[indices]=mu_err+sigma_err*np.random.randn(nbr_err)
        
    # Mesures bruitees
    
    y_bruit=y_vrai+bruit
    
    # Renvoi des instants de mesure et des mesures bruitees
    
    return(x_mes,y_bruit)
    
# Fonction noyau moindres carres
# En entree,
#   - matA      = matrice modele A du systeme d'observations
#   - vecB      = vecteur d'observations B du systeme d'observations
#   - matP      = matrice de poids P de l'estimation
#   - choix_mes = vecteur des indices des observations a conserver
#
# En sortie,
#   - Xchap     = vecteur des estimations des parametres par moindres carres
#   - Vchap     = vecteur des residus de l'estimation par moindres carres
    
def moindres_carres(matA,vecB,matP,choix_mes):
    
    # Sous-systeme d'observations
    
    matA_choix=matA[choix_mes,:]
    vecB_choix=vecB[choix_mes,:]
    
    matP_choix=matP[np.ix_(choix_mes,choix_mes)]
    
    # Systeme normal
    
    matN=matA_choix.T.dot(matP_choix.dot(matA_choix))
    vecK=matA_choix.T.dot(matP_choix.dot(vecB_choix))
    
    # Estimation par moindres carres
    
    Xchap=np.linalg.inv(matN).dot(vecK)
    
    # Vecteur des residus pour l'ensemble des mesures
    
    Vchap=vecB-matA.dot(Xchap)
    
    # Renvoi des parametres estimes et des residus
    
    return Xchap,Vchap

