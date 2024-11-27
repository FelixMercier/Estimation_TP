import numpy as np
import matplotlib.pyplot as plt
import Fonctions_TP as tp
from scipy.optimize import linprog
from scipy.stats import norm

######

def w_h(V, c):
    w = np.zeros(V.shape)
    for i in range(len(w)):
        if np.abs(V[i]) <= c:
            w[i] = 1
        else:
            w[i] = c/np.abs(V[i])
    return w
    
def w_t(V, c):
    med_v = np.median(np.abs(V))
    res = [V[i]/(6*med_v) for i in range(len(V))]
    w = np.zeros(V.shape)
    for i in range(len(w)):
        if np.abs(res[i]) <= 1:
            w[i] = (1-res[i]**2)**2
    return w

def ransac(A, B, P, n, t, T, K):
    nb_bonnes_mes_abs = [1]
    choix_mes_abs = np.array([[0], [0]])
    
    for i in range(K):
        
        choixmes = np.random.choice(100, n, replace = False)
        
        Xchap, Vchap = tp.moindres_carres(A, B, P, choixmes)
        
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


def MPCI(eps, w_t_or_h, A, B, c, k_max):
    """
    eps : seuil
    w : w_t pout tukey, w_h pour huber
    """
    #initalisation
    P = np.eye(len(A))
    N = A.T @ P @ A
    K = A.T @ P @ B
    Xchap = np.linalg.inv(N) @ K
    Vchap = B - A @ Xchap
    w = w_t_or_h(Vchap, c)
   
    k = 0
    while k < k_max:
        k+=1
        
        Xchap_km1 = Xchap
        P = np.diag(w.reshape(1, -1)[0])
        N = A.T @ P @ A
        K = A.T@ P @ B
        Xchap = np.linalg.inv(N) @ K
        Vchap = B - A @ Xchap
        w = w_t_or_h(Vchap, c)
        
        num = np.abs(Xchap - Xchap_km1)
        Vec_test = [num[i]/np.abs(Xchap_km1[i]) for i in range(len(num))]
        for val in Vec_test:
            if val > eps: break
            return Xchap, Vchap, k
    return Xchap, Vchap, k

        
if __name__ == '__main__' : 
    
    xmes, ybruit = tp.creer_mesures(2, 5, -50, 50, 100, 1, 0, 0, 0)
    vecx = xmes.reshape(len(xmes), 1)
    vecy = ybruit.reshape(len(ybruit), 1)
    
    estimation_method = 'tukey'
    
    if estimation_method == 'ransac':
        
        A = np.hstack(( vecx, np.ones(np.shape(vecx)) ))

        B = np.abs(vecy)

        sigma = 5
        P = sigma*np.eye(len(vecx))

        print('Estimation par ransac')
        
        Xchap, Vchap, n_iter = ransac(A, vecy, P, 2, 1.96, 95, 100)
        a, b = Xchap[0, 0], Xchap[1, 0]
        print(f"Estimation par la méthode de RANSAC :")
        print(f"Valeurs trouvée pour les deux paramètres :")
        print(f"a = {np.round(Xchap[0][0], 5)} et b = {np.round(Xchap[1][0], 5)}")

        esti_ransac = plt.figure()
        plt.scatter(vecx, vecy, c='blue', s=10, label='données bruitées')
        plt.plot(vecx, a*vecx + b, c='r', label='droite des paramètres estimés')
        plt.title("Résultat de l'estimation par la méthode de RANSAC")
        plt.legend()
        plt.show()
        
        res = plt.figure()
        plt.scatter(vecx, Vchap, c='blue', s=15, label="Résidus")
        plt.title("Résidus de l'estimation par la méthode de RANSAC")
        plt.legend()
        plt.show()
        
        
    if estimation_method == 'huber':
        A = np.hstack((vecx, np.ones(vecx.shape)))
        B = vecy
        c=3
        
        Xchap, Vchap, iterations = MPCI(0.000001, w_h, A, B, c, 50)
        a, b = Xchap[0, 0], Xchap[1, 0]
        print(f"Estimation par MPCI-huber :")
        print(f"Valeurs trouvée pour les deux paramètres en {iterations} itérations :")
        print(f"a = {np.round(Xchap[0][0], 5)} et b = {np.round(Xchap[1][0], 5)}")
        
        est = plt.figure()
        plt.scatter(vecx, vecy, c='blue', s=10, label='données bruitées')
        plt.plot(vecx, a*vecx + b, c='r', label='droite des paramètres estimés')
        plt.title("Résultat de l'estimation par moindre carrés pondérés avec la méthode de Huber")
        plt.legend()
        plt.show()
        
        res = plt.figure()
        plt.scatter(vecx, Vchap, c='blue', s=15, label="Résidus")
        plt.title("Résidus de l'estimation par moindres carés pondérés avec la méthode de Huber")
        plt.legend()
        plt.show()
        
        
    
    if estimation_method == 'tukey':
        A = np.hstack((vecx, np.ones(vecx.shape)))
        B = vecy
        c=3
        
        Xchap, Vchap, iterations = MPCI(0.000001, w_t, A, B, c, 50)
        a, b = Xchap[0, 0], Xchap[1, 0]
        print(f"Estimation par MPCI-Tukey :")
        print(f"Valeurs trouvée pour les deux paramètres en {iterations} itérations :")
        print(f"a = {np.round(Xchap[0][0], 5)} et b = {np.round(Xchap[1][0], 5)}")
        
        est = plt.figure()
        plt.scatter(vecx, vecy, c='blue', s=10, label='données bruitées')
        plt.plot(vecx, a*vecx + b, c='r', label='droite des paramètres estimés')
        plt.title("Résultat de l'estimation par moindre carrés pondérés avec la méthode de Tukey")
        plt.legend()
        plt.show()
        
        res = plt.figure()
        plt.scatter(vecx, Vchap, c='blue', s=15, label="Résidus")
        plt.title("Résidus de l'estimation par moindres carés pondérés avec la méthode de Tukey")
        plt.legend()
        plt.show()
        
        
    PAR_huber = np.zeros((100, 2))
    PAR_tukey = np.zeros((100, 2))
    PAR_ransac = np.zeros((100, 2))
    
    ITER_huber = np.zeros((100, 1))
    ITER_tukey = np.zeros((100, 1))
    ITER_ransac = np.zeros((100, 1))
    
    RES_moy_huber = np.zeros((100, 1))
    RES_moy_tukey = np.zeros((100, 1))
    RES_moy_ransac = np.zeros((100, 1))
    RES_std_huber = np.zeros((100, 1))
    RES_std_tukey = np.zeros((100, 1))
    RES_std_ransac = np.zeros((100, 1))
        
    #%%
    #generer 100 jeu de données
    for i in range(100):
        xmes, ybruit = tp.creer_mesures(2, 5, -50, 50, 100, 1, 0, 0, 0)
        vecx = xmes.reshape(len(xmes), 1)
        vecy = ybruit.reshape(len(ybruit), 1)
        
        Xchap_huber , Vchap_huber, iter_huber = MPCI(0.000001, w_h, A, B, c, 50)
        Xchap_tukey , Vchap_tukey, iter_tukey = MPCI(0.000001, w_t, A, B, c, 50)
        Xchap_ransac , Vchap_ransac, iter_ransac = ransac(A, vecy, P, 2, 1.96, 95, 100)
        
        PAR_huber[i, 0], PAR_huber[i, 1] = np.abs(2-Xchap_huber[0, 0]), np.abs(5-Xchap_huber[1, 0])
        PAR_tukey[i, 0], PAR_tukey[i, 1] = np.abs(2-Xchap_tukey[0, 0]), np.abs(5-Xchap_tukey[1, 0])
        PAR_ransac[i, 0], PAR_ransac[i, 1] = np.abs(2-Xchap_ransac[0, 0]), np.abs(5-Xchap_ransac[1, 0])
        
        ITER_huber[i, 0] = iter_huber
        ITER_tukey[i, 0] = iter_tukey
        ITER_ransac[i, 0] = iter_ransac
        
        RES_moy_huber[i] = np.mean(Vchap_huber)
        RES_moy_tukey[i] = np.mean(Vchap_tukey)
        RES_moy_ransac[i] = np.mean(Vchap_ransac)
        RES_std_huber[i] = np.std(Vchap_huber)
        RES_std_tukey[i] = np.std(Vchap_tukey)
        RES_std_ransac[i] = np.std(Vchap_ransac)
        
    a_mean_huber = np.mean(PAR_huber[:, 0])
    b_mean_huber = np.mean(PAR_huber[:, 1])
    a_mean_tukey = np.mean(PAR_tukey[:, 0])
    b_mean_tukey = np.mean(PAR_tukey[:, 1])
    a_mean_ransac = np.mean(PAR_ransac[:, 0])
    b_mean_ransac = np.mean(PAR_ransac[:, 1])
    a_std_huber = np.std(PAR_huber[:, 0])
    b_std_huber = np.std(PAR_huber[:, 1])
    a_std_tukey = np.std(PAR_tukey[:, 0])
    b_std_tukey = np.std(PAR_tukey[:, 1])
    a_std_ransac = np.std(PAR_ransac[:, 0])
    b_std_ransac = np.std(PAR_ransac[:, 1])
    
    moy_it_huber = np.mean(ITER_huber)
    moy_it_tukey = np.mean(ITER_tukey)
    moy_it_ransac = np.mean(ITER_ransac)
    
    moy_res_huber = np.mean(RES_moy_huber)
    std_med_huber = np.median(RES_std_huber)
    moy_res_tukey = np.mean(RES_moy_tukey)
    std_med_tukey = np.median(RES_std_tukey)
    moy_res_ransac = np.mean(RES_moy_ransac)
    std_med_ransac = np.median(RES_std_ransac)
    
        
    print("Huber : ")
    print(f"a = {a_mean_huber} +/- {a_std_huber}")
    print(f"b = {b_mean_huber} +/- {b_std_huber}")
    print(f"en moyenne, en {moy_it_huber} itérations")
    print(f"En moyenne les résidus sont de {moy_res_huber} +/- {std_med_huber}")
    
    print(" ")
    
    print("Tukey : ")
    print(f"a = {a_mean_tukey} +/- {a_std_tukey}")
    print(f"b = {b_mean_tukey} +/- {b_std_tukey}")
    print(f"en moyenne, en {moy_it_tukey} itérations")
    print(f"En moyenne les résidus sont de {moy_res_tukey} +/- {std_med_tukey}")
    
    print(" ")
    
    print("Ransac : ")
    print(f"a = {a_mean_ransac} +/- {a_std_ransac}")
    print(f"b = {b_mean_ransac} +/- {b_std_ransac}")
    print(f"en moyenne, en {moy_it_tukey} itérations")
    print(f"En moyenne les résidus sont de {moy_res_ransac} +/- {std_med_ransac}")
    
    for i in range(100):
        xmes, ybruit = tp.creer_mesures(2, 5, -50, 50, 100, 1, 5, 10, 1)
        vecx = xmes.reshape(len(xmes), 1)
        vecy = ybruit.reshape(len(ybruit), 1)
        
        Xchap_huber , Vchap_huber, iter_huber = MPCI(0.000001, w_h, A, B, c, 50)
        Xchap_tukey , Vchap_tukey, iter_tukey = MPCI(0.000001, w_t, A, B, c, 50)
        Xchap_ransac , Vchap_ransac, iter_ransac = ransac(A, vecy, P, 2, 1.96, 95, 100)
        
        PAR_huber[i, 0], PAR_huber[i, 1] = np.abs(2-Xchap_huber[0, 0]), np.abs(5-Xchap_huber[1, 0])
        PAR_tukey[i, 0], PAR_tukey[i, 1] = np.abs(2-Xchap_tukey[0, 0]), np.abs(5-Xchap_tukey[1, 0])
        PAR_ransac[i, 0], PAR_ransac[i, 1] = np.abs(2-Xchap_ransac[0, 0]), np.abs(5-Xchap_ransac[1, 0])
        
        ITER_huber[i, 0] = iter_huber
        ITER_tukey[i, 0] = iter_tukey
        ITER_ransac[i, 0] = iter_ransac
        
        RES_moy_huber[i] = np.mean(Vchap_huber)
        RES_moy_tukey[i] = np.mean(Vchap_tukey)
        RES_moy_ransac[i] = np.mean(Vchap_ransac)
        RES_std_huber[i] = np.std(Vchap_huber)
        RES_std_tukey[i] = np.std(Vchap_tukey)
        RES_std_ransac[i] = np.std(Vchap_ransac)
        
    a_mean_huber = np.mean(PAR_huber[:, 0])
    b_mean_huber = np.mean(PAR_huber[:, 1])
    a_mean_tukey = np.mean(PAR_tukey[:, 0])
    b_mean_tukey = np.mean(PAR_tukey[:, 1])
    a_mean_ransac = np.mean(PAR_ransac[:, 0])
    b_mean_ransac = np.mean(PAR_ransac[:, 1])
    a_std_huber = np.std(PAR_huber[:, 0])
    b_std_huber = np.std(PAR_huber[:, 1])
    a_std_tukey = np.std(PAR_tukey[:, 0])
    b_std_tukey = np.std(PAR_tukey[:, 1])
    a_std_ransac = np.std(PAR_ransac[:, 0])
    b_std_ransac = np.std(PAR_ransac[:, 1])
    
    moy_it_huber = np.mean(ITER_huber)
    moy_it_tukey = np.mean(ITER_tukey)
    moy_it_ransac = np.mean(ITER_ransac)
    
    moy_res_huber = np.mean(RES_moy_huber)
    std_med_huber = np.median(RES_std_huber)
    moy_res_tukey = np.mean(RES_moy_tukey)
    std_med_tukey = np.median(RES_std_tukey)
    moy_res_ransac = np.mean(RES_moy_ransac)
    std_med_ransac = np.median(RES_std_ransac)
    
    print("______________________________________________")
    print(" ")
    print(" ")
    print("5% d'erreur, N(10, 1)")
    print(" ")
        
    print("Huber : ")
    print(f"a = {a_mean_huber} +/- {a_std_huber}")
    print(f"b = {b_mean_huber} +/- {b_std_huber}")
    print(f"en moyenne, en {moy_it_huber} itérations")
    print(f"En moyenne les résidus sont de {moy_res_huber} +/- {std_med_huber}")
    
    print(" ")
    
    print("Tukey : ")
    print(f"a = {a_mean_tukey} +/- {a_std_tukey}")
    print(f"b = {b_mean_tukey} +/- {b_std_tukey}")
    print(f"en moyenne, en {moy_it_tukey} itérations")
    print(f"En moyenne les résidus sont de {moy_res_tukey} +/- {std_med_tukey}")
    
    print(" ")
    
    print("Ransac : ")
    print(f"a = {a_mean_ransac} +/- {a_std_ransac}")
    print(f"b = {b_mean_ransac} +/- {b_std_ransac}")
    print(f"en moyenne, en {moy_it_tukey} itérations")
    print(f"En moyenne les résidus sont de {moy_res_ransac} +/- {std_med_ransac}")
    print("______________________________________________")
    
    for i in range(100):
        xmes, ybruit = tp.creer_mesures(2, 5, -50, 50, 100, 1, 10, 10, 1)
        vecx = xmes.reshape(len(xmes), 1)
        vecy = ybruit.reshape(len(ybruit), 1)
        
        Xchap_huber , Vchap_huber, iter_huber = MPCI(0.000001, w_h, A, B, c, 50)
        Xchap_tukey , Vchap_tukey, iter_tukey = MPCI(0.000001, w_t, A, B, c, 50)
        Xchap_ransac , Vchap_ransac, iter_ransac = ransac(A, vecy, P, 2, 1.96, 95, 100)
        
        PAR_huber[i, 0], PAR_huber[i, 1] = np.abs(2-Xchap_huber[0, 0]), np.abs(5-Xchap_huber[1, 0])
        PAR_tukey[i, 0], PAR_tukey[i, 1] = np.abs(2-Xchap_tukey[0, 0]), np.abs(5-Xchap_tukey[1, 0])
        PAR_ransac[i, 0], PAR_ransac[i, 1] = np.abs(2-Xchap_ransac[0, 0]), np.abs(5-Xchap_ransac[1, 0])
        
        ITER_huber[i, 0] = iter_huber
        ITER_tukey[i, 0] = iter_tukey
        ITER_ransac[i, 0] = iter_ransac
        
        RES_moy_huber[i] = np.mean(Vchap_huber)
        RES_moy_tukey[i] = np.mean(Vchap_tukey)
        RES_moy_ransac[i] = np.mean(Vchap_ransac)
        RES_std_huber[i] = np.std(Vchap_huber)
        RES_std_tukey[i] = np.std(Vchap_tukey)
        RES_std_ransac[i] = np.std(Vchap_ransac)
        
    a_mean_huber = np.mean(PAR_huber[:, 0])
    b_mean_huber = np.mean(PAR_huber[:, 1])
    a_mean_tukey = np.mean(PAR_tukey[:, 0])
    b_mean_tukey = np.mean(PAR_tukey[:, 1])
    a_mean_ransac = np.mean(PAR_ransac[:, 0])
    b_mean_ransac = np.mean(PAR_ransac[:, 1])
    a_std_huber = np.std(PAR_huber[:, 0])
    b_std_huber = np.std(PAR_huber[:, 1])
    a_std_tukey = np.std(PAR_tukey[:, 0])
    b_std_tukey = np.std(PAR_tukey[:, 1])
    a_std_ransac = np.std(PAR_ransac[:, 0])
    b_std_ransac = np.std(PAR_ransac[:, 1])
    
    moy_it_huber = np.mean(ITER_huber)
    moy_it_tukey = np.mean(ITER_tukey)
    moy_it_ransac = np.mean(ITER_ransac)
    
    moy_res_huber = np.mean(RES_moy_huber)
    std_med_huber = np.median(RES_std_huber)
    moy_res_tukey = np.mean(RES_moy_tukey)
    std_med_tukey = np.median(RES_std_tukey)
    moy_res_ransac = np.mean(RES_moy_ransac)
    std_med_ransac = np.median(RES_std_ransac)
    
    print("______________________________________________")
    print(" ")
    print(" ")
    
    print("10% d'erreur, N(10, 1)")
    print(" ")
    
        
    print("Huber : ")
    print(f"a = {a_mean_huber} +/- {a_std_huber}")
    print(f"b = {b_mean_huber} +/- {b_std_huber}")
    print(f"en moyenne, en {moy_it_huber} itérations")
    print(f"En moyenne les résidus sont de {moy_res_huber} +/- {std_med_huber}")
    
    print(" ")
    
    print("Tukey : ")
    print(f"a = {a_mean_tukey} +/- {a_std_tukey}")
    print(f"b = {b_mean_tukey} +/- {b_std_tukey}")
    print(f"en moyenne, en {moy_it_tukey} itérations")
    print(f"En moyenne les résidus sont de {moy_res_tukey} +/- {std_med_tukey}")
    
    print(" ")
    
    print("Ransac : ")
    print(f"a = {a_mean_ransac} +/- {a_std_ransac}")
    print(f"b = {b_mean_ransac} +/- {b_std_ransac}")
    print(f"en moyenne, en {moy_it_tukey} itérations")
    print(f"En moyenne les résidus sont de {moy_res_ransac} +/- {std_med_ransac}")
    print("______________________________________________")

    
    for i in range(100):
        xmes, ybruit = tp.creer_mesures(2, 5, -50, 50, 100, 1, 25, 10, 1)
        vecx = xmes.reshape(len(xmes), 1)
        vecy = ybruit.reshape(len(ybruit), 1)
        
        Xchap_huber , Vchap_huber, iter_huber = MPCI(0.000001, w_h, A, B, c, 50)
        Xchap_tukey , Vchap_tukey, iter_tukey = MPCI(0.000001, w_t, A, B, c, 50)
        Xchap_ransac , Vchap_ransac, iter_ransac = ransac(A, vecy, P, 2, 1.96, 95, 100)
        
        PAR_huber[i, 0], PAR_huber[i, 1] = np.abs(2-Xchap_huber[0, 0]), np.abs(5-Xchap_huber[1, 0])
        PAR_tukey[i, 0], PAR_tukey[i, 1] = np.abs(2-Xchap_tukey[0, 0]), np.abs(5-Xchap_tukey[1, 0])
        PAR_ransac[i, 0], PAR_ransac[i, 1] = np.abs(2-Xchap_ransac[0, 0]), np.abs(5-Xchap_ransac[1, 0])
        
        ITER_huber[i, 0] = iter_huber
        ITER_tukey[i, 0] = iter_tukey
        ITER_ransac[i, 0] = iter_ransac
        
        RES_moy_huber[i] = np.mean(Vchap_huber)
        RES_moy_tukey[i] = np.mean(Vchap_tukey)
        RES_moy_ransac[i] = np.mean(Vchap_ransac)
        RES_std_huber[i] = np.std(Vchap_huber)
        RES_std_tukey[i] = np.std(Vchap_tukey)
        RES_std_ransac[i] = np.std(Vchap_ransac)
        
    a_mean_huber = np.mean(PAR_huber[:, 0])
    b_mean_huber = np.mean(PAR_huber[:, 1])
    a_mean_tukey = np.mean(PAR_tukey[:, 0])
    b_mean_tukey = np.mean(PAR_tukey[:, 1])
    a_mean_ransac = np.mean(PAR_ransac[:, 0])
    b_mean_ransac = np.mean(PAR_ransac[:, 1])
    a_std_huber = np.std(PAR_huber[:, 0])
    b_std_huber = np.std(PAR_huber[:, 1])
    a_std_tukey = np.std(PAR_tukey[:, 0])
    b_std_tukey = np.std(PAR_tukey[:, 1])
    a_std_ransac = np.std(PAR_ransac[:, 0])
    b_std_ransac = np.std(PAR_ransac[:, 1])
    
    moy_it_huber = np.mean(ITER_huber)
    moy_it_tukey = np.mean(ITER_tukey)
    moy_it_ransac = np.mean(ITER_ransac)
    
    moy_res_huber = np.mean(RES_moy_huber)
    std_med_huber = np.median(RES_std_huber)
    moy_res_tukey = np.mean(RES_moy_tukey)
    std_med_tukey = np.median(RES_std_tukey)
    moy_res_ransac = np.mean(RES_moy_ransac)
    std_med_ransac = np.median(RES_std_ransac)
    
    print("______________________________________________")
    print(" ")
    print(" ")
        
    print("25% d'erreur, N(10, 1)")
    print(" ")
    
    print("Huber : ")
    print(f"a = {a_mean_huber} +/- {a_std_huber}")
    print(f"b = {b_mean_huber} +/- {b_std_huber}")
    print(f"en moyenne, en {moy_it_huber} itérations")
    print(f"En moyenne les résidus sont de {moy_res_huber} +/- {std_med_huber}")
    
    print(" ")
    
    print("Tukey : ")
    print(f"a = {a_mean_tukey} +/- {a_std_tukey}")
    print(f"b = {b_mean_tukey} +/- {b_std_tukey}")
    print(f"en moyenne, en {moy_it_tukey} itérations")
    print(f"En moyenne les résidus sont de {moy_res_tukey} +/- {std_med_tukey}")
    
    print(" ")
    
    print("Ransac : ")
    print(f"a = {a_mean_ransac} +/- {a_std_ransac}")
    print(f"b = {b_mean_ransac} +/- {b_std_ransac}")
    print(f"en moyenne, en {moy_it_tukey} itérations")
    print(f"En moyenne les résidus sont de {moy_res_ransac} +/- {std_med_ransac}")
    print("______________________________________________")

    
    
    
        
#%%
    
        
        
        
        
        
        
        
        

        
                
        
        
        
        
        
        

