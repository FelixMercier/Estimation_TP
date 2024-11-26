import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog
from scipy.stats import norm

def read_dat(filename, header=0):

    with open(filename, 'r') as file:
        id_line = 0
        for line in file:
            id_line += 1
            row = line.split()
            if id_line <= header: continue
        
            elif id_line == 2:
                data = np.array([float(x) for x in row])
                continue
            
            data = np.vstack((data, np.array([float(x) for x in row])))
        return data
    
mes_noisy = read_dat("mesures_bruitees.dat", header=1)
t = mes_noisy[:, 0].reshape(len(mes_noisy), 1)
E_bruit = mes_noisy[:, 1].reshape(len(mes_noisy), 1)
N_bruit = mes_noisy[:, 2].reshape(len(mes_noisy), 1)
sigE = mes_noisy[:, 3].reshape(len(mes_noisy), 1)
sigN = mes_noisy[:, 4].reshape(len(mes_noisy), 1)

mes_vraies = read_dat("mesures_vraies.dat", header=1)
E_vraies = mes_vraies[:, 1].reshape(len(mes_noisy), 1)
N_vraies = mes_vraies[:, 2].reshape(len(mes_noisy), 1)

dt = 0.5

# traj = plt.figure()
# plt.plot(E_vraies, N_vraies, c='r', label='vraie trajectoire')
# plt.plot(E_bruit, N_bruit, c='blue', label='trajectoire bruitée', alpha=0.5)
# plt.title("Visualisation des deux trajectoires")
# plt.legend()
# plt.show()


A = np.eye(4)
A[0, 2] += dt
A[1, 3] += dt

C = np.hstack((np.eye(2), np.zeros((2, 2))))

def Rt(t):
    return np.array([[sigE[t, 0]**2, 0],
                     [0, sigN[t, 0]**2]])



def Yt(t):
    return mes_noisy[t, 1:3].reshape(2, 1)


def Kalman(sigma_0, p):
    """p : nb_para
       sigmamod : valeur de sigmamod dans Qt
    """
    X_t_tm1_chap = np.vstack((Yt(0), np.array([[0], [0]]))) #valeur initiale
    Sigma_t_tm1 = np.eye(4) * sigma_0 #valeur initiale
    
    Kt = Sigma_t_tm1 @ C.T @ np.linalg.inv(C @ Sigma_t_tm1 @ C.T + Rt(0))
    X_t_t_chap = X_t_tm1_chap + Kt @ (Yt(0) - C @ X_t_tm1_chap)  ######!!!!!!!!!!!!!pas sur du 0 !!!!!!!!!!
    Sigma_t_t = (np.eye(p) - Kt @ C) @ Sigma_t_tm1
    
    X_updt = np.array([x for x in X_t_t_chap]).reshape((1, 4))
    X_pred = np.array([x for x in X_t_tm1_chap]).reshape((1, 4))
    
    Sig_updt = np.zeros((len(mes_noisy), np.shape(Sigma_t_t)[0], np.shape(Sigma_t_t)[1] ))
    Sig_updt[0] = Sigma_t_t
    Sig_pred = np.zeros((len(mes_noisy), np.shape(Sigma_t_t)[0], np.shape(Sigma_t_t)[1] ))
    Sig_pred[0] = Sigma_t_tm1
    
    for i in range(1, len(mes_noisy)):
    
        #Prédiction :
        X_tp1_t_chap = A @ X_t_t_chap
        Sigma_tp1_t = A@Sigma_t_t@A.T + Qt
               
        #MAJ
        Kt = Sigma_tp1_t @ C.T @ np.linalg.inv(C @ Sigma_tp1_t @ C.T + Rt(i))
        X_tp1_tp1_chap = X_tp1_t_chap + Kt @ (Yt(i) - C @ X_tp1_t_chap)
        Sigma_tp1_tp1 = (np.eye(p) - Kt @ C) @ Sigma_tp1_t
        
        X_t_tm1_chap = X_tp1_t_chap
        Sigma_t_tm1 = Sigma_tp1_t
        X_t_t_chap = X_tp1_tp1_chap  #résultat de l'analyse
        Sigma_t_t = Sigma_tp1_tp1    #résultat de l'analyse
        
        X_updt = np.vstack((X_updt, np.array([x for x in X_t_t_chap]).reshape((1, 4)) ))
        X_pred = np.vstack((X_pred, np.array([x for x in X_t_tm1_chap]).reshape((1, 4)) ))
        
        Sig_updt[i] = Sigma_t_t
        Sig_pred[i] = Sigma_t_tm1
        
    return X_updt, X_pred, Sig_updt, Sig_pred
        
sigmamod=1
Qt = np.zeros((4,4))
Qt[2, 2] += sigmamod**2
Qt[3, 3] += sigmamod**2
        
X_updt, X_pred, Sig_updt, Sig_pred = Kalman(10, 4)

E_est = X_updt[:, 0]
N_est = X_updt[:, 1]
Vit_E = X_updt[:, 2]
Vit_N = X_updt[:, 3]

def lissage(X_updt, Sig_updt):
    X_smooth = X_updt.copy()
    Sig_smooth = Sig_updt.copy()
    
    for i in range(len(mes_noisy)-2, -1, -1):
        X_up = X_updt[i].reshape(-1, 1)
        
        Xf = X_pred[i+1].reshape(-1, 1)
        Sig_up = Sig_updt[i]
        Sigf = Sig_pred[i]
        
        F_t= Sig_up @ A.T @ np.linalg.inv(Sigf)
        x_smth = X_up + F_t @ (X_smooth[i+1].reshape(-1, 1) - Xf)
        X_smooth[i] = x_smth.reshape(1, -1)[0]
        Sig_smooth[i] = Sig_up + F_t @ (Sig_smooth[i+1] - Sig_pred[i+1]) @ F_t.T
        
    return X_smooth, Sig_smooth

X_liss, Sig_liss = lissage(X_updt, Sig_updt)

E_liss = X_liss[:, 0]
N_liss = X_liss[:, 1]
E_vit_liss = X_liss[:, 2]
N_vit_liss = X_liss[:, 3]

traj = plt.figure()
plt.plot(E_vraies, N_vraies, c='black', label='vraie trajectoire')
plt.plot(E_bruit, N_bruit, c='blue', label='trajectoire bruitée', alpha=0.5)
# plt.plot(E_est, N_est, c='red', label='trajectoire estimée', alpha=0.6)
plt.plot(E_liss, N_liss, c="red", label='lisée', alpha = 0.6)
plt.title("Visualisation des deux trajectoires, sigma_mod = {}".format(sigmamod))
plt.legend()
plt.show()
