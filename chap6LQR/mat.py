import numpy as np
import scipy.linalg
import sys
sys.path.append('..')
from tools.tools import Quaternion2Euler
import pandas as pd
pd.set_option('display.width', 320)
pd.set_option('display.max_columns', 12)
np.set_printoptions(linewidth=320)



def lqr(A,B,Q,R):
    X = np.matrix(scipy.linalg.solve_continuous_are(A,B,Q,R))
    K = np.matrix(scipy.linalg.inv(R)*(B.T*X))
    #eigenVals, eigenVect = scipy.linalg.eig(A-B*K)
    return K


trim_input = np.array([[-1.24785989e-01],
                      [ 3.14346798e-01],
                      [-8.13462186e-09],
                      [-1.21428507e-08]])

de_eq = trim_input.item(0)
dt_eq = trim_input.item(1)
da_eq = trim_input.item(2)
dr_eq = trim_input.item(3)

trim_state = np.array([[-5.63618896e-15],
                       [ 0.00000000e+00],
                       [-1.00000000e+02],
                       [ 2.49687391e+01],
                       [ 0.00000000e+00],
                       [ 1.24982663e+00],
                       [ 9.99687343e-01],
                       [ 0.00000000e+00],
                       [ 2.50043383e-02],
                       [ 0.00000000e+00],
                       [ 0.00000000e+00],
                       [ 0.00000000e+00],
                       [ 0.00000000e+00]])

e = np.array([trim_state.item(6),trim_state.item(7),trim_state.item(8),trim_state.item(9)])
[phi_eq, theta_eq, psi_eq] = Quaternion2Euler(e)

#### State Space with V
# A_lat = np.array([[-7.76772583e-01,  1.24982663e+00, -2.49687391e+01,  9.79995917e+00, -1.70856859e-16],
#                  [-3.86674768e+00, -2.26288507e+01,  1.09050408e+01,  0.00000000e+00,  0.00000000e+00],
#                  [ 7.83075106e-01, -1.15091677e-01, -1.22765474e+00,  0.00000000e+00,  0.00000000e+00],
#                  [ 0.00000000e+00,  9.99966585e-01,  5.00522872e-02,  0.00000000e+00,  0.00000000e+00],
#                  [ 0.00000000e+00, -1.67040911e-06,  1.00121846e+00, 0.00000000e+00,  0.00000000e+00]])
# B_lat = np.array([[  1.48617188,   3.76496875],
#                  [130.8836782,   -1.79637437],
#                  [  5.01173501, -24.88134133],
#                  [  0.,           0.        ],
#                  [  0.,           0.        ]])
#
# C_lat = np.array([[1., 0., 0., 0., 0.],#, 0., 0.],
#                   [0., 0., 0., 0., 1.]])#, 0., 0.]])
#
# Klat = np.array([[1.6806e-01,   8.5475e-01,   8.9091e-02,   1.7618e+00,   3.1464e+00],
#                  [9.7195e-01,   2.3630e-02,  -1.5278e+00,   4.7121e-01,  -3.7471e-01]])
#
# Kilat = np.array([[-1.6019e-01, -9.8709e-01],
#                   [-9.8709e-01,  1.6019e-01]])

####State Space with Beta instead of v
A_lat = np.array([[-7.76772583e-01,  1.24982663e+00/25., -2.49687391e+01/25.,  9.79995917e+00/25., -1.70856859e-16/25.],
                 [-3.86674768e+00*25., -2.26288507e+01,  1.09050408e+01,  0.00000000e+00,  0.00000000e+00],
                 [ 7.83075106e-01*25., -1.15091677e-01, -1.22765474e+00,  0.00000000e+00,  0.00000000e+00],
                 [ 0.00000000e+00,  9.99966585e-01,  5.00522872e-02,  0.00000000e+00,  0.00000000e+00],
                 [ 0.00000000e+00, -1.67040911e-06,  1.00121846e+00, 0.00000000e+00,  0.00000000e+00]])
B_lat = np.array([[  1.48617188,   3.76496875],
                 [130.8836782,   -1.79637437],
                 [  5.01173501, -24.88134133],
                 [  0.,           0.        ],
                 [  0.,           0.        ]])

C_lat = np.array([[1., 0., 0., 0., 0.],#, 0., 0.],
                  [0., 0., 0., 0., 1.]])#, 0., 0.]])

Alataug_T = np.hstack((A_lat,np.zeros((5,2))))

Cr = np.array([[-1., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., -1., 0., 0.]])

Alat_aug = np.vstack((Alataug_T,Cr))

Blat_aug = np.vstack((B_lat,np.zeros((2,2))))

# Clat_aug = np.array[-1 0 0 0 0 0 0;
#             0 0 0 0 -1 0 0];

betamax = np.pi/3.
pmax = 1.
rmax = 1.
phimax = np.pi/3.
psimax = np.pi/3.
beta_int_max = 3
psi_int_max = 5

Qlat = np.array([[1./betamax**2., 0., 0., 0., 0., 0., 0.],
                 [0., 1./pmax**2, 0., 0., 0., 0., 0.],
                 [0., 0., 1./rmax**2., 0., 0., 0., 0.],
                 [0., 0., 0., 1./phimax**2., 0., 0., 0.],
                 [0., 0., 0., 0., 1./psimax**2., 0., 0.],
                 [0., 0., 0., 0., 0., 1./beta_int_max**2., 0.],
                 [0., 0., 0., 0., 0., 0., 1./psi_int_max**2.]])


damax = np.radians(01.)
drmax = np.radians(01.)
Rlat = np.array([[1./damax**2, 0.],[0., 1./drmax**2]])

Klat_all = lqr(Alat_aug,Blat_aug,Qlat,Rlat);

Klat = Klat_all[:,0:5]
Kilat = Klat_all[:,5:]
# Klat = np.array([[2.3297e-03,   1.0044e-01,  -1.3544e+00,  -2.4719e+01,   1.7060e+00],
#                  [1.0098e+00,  -9.1437e-02,  -4.0277e-04,   2.5987e+00,  -2.5730e-01]])
#
# Kilat = np.array([[-1.6348e-01,  -9.8655e-01],
#                   [-9.8655e-01,   1.6348e-01]])

xlat_eq = np.array([[0., 0., 0., phi_eq, psi_eq]]).T

ylat_eq = np.array([[0.,0.]]).T

ulat_eq = np.array([[da_eq, dr_eq]]).T

limitlat = np.array([[np.radians(45.), np.radians(45.)]])


### Lon with w
# A_lon = np.array([[ -0.57681981,   0.48178674,  -1.21990873,  -9.78648014,   0.        ],
#                  [ -0.56064823,  -4.46355336,  24.37104644,  -0.56472816,   0.        ],
#                  [  0.19994698,  -3.99297803,  -5.2947383,   0.,           0.        ],
#                  [  0.,           0.,           0.99971072,   0.,           0.        ],
#                  [ 0.04999304,   -0.99874957,   0.,         25.00631447,   0.        ]])
#
# B_lon = np.array([[ -0.13839273,  47.76297312],
#                  [ -2.58618378,   0.        ],
#                  [-36.11238957,   0.        ],
#                  [  0.,           0.        ],
#                  [  0.,           0.        ]])
#
# C_lon = np.array([[1., 0., 0., 0., 0.],#, 0., 0.],
#                   [0., 0., 0., 0., 1.]])#, 0., 0.]])

### Lon with alpha
A_lon = np.array([[ -0.57681981,   0.48178674*25.,  -1.21990873,  -9.81*np.cos(theta_eq), 0.],#-9.78648014,   0.        ],
                 [ -0.56064823/25.,  -4.46355336,  24.37104644/25.,  -0.56472816/25.,   0.        ],
                 [  0.19994698,  -3.99297803*25.,  -5.2947383,   0.,           0.        ],
                 [  0.,           0.,           1., 0.,0.],#0.99971072,   0.,           0.        ],
                 [ np.sin(theta_eq),   -np.cos(theta_eq)*25., 0., trim_state.item(3)*np.cos(theta_eq)+trim_state.item(5)*np.sin(theta_eq), 0.]])

B_lon = np.array([[ -0.13839273,  47.015150132144704],#47.76297312],
                 [ -2.58618378/25.,   0.        ],
                 [-36.11238957,   0.        ],
                 [  0.,           0.        ],
                 [  0.,           0.        ]])

C_lon = np.array([[1., 0., 0., 0., 0.],#, 0., 0.],
                  [0., 0., 0., 0., 1.]])#, 0., 0.]])

Alonaug_T = np.hstack((A_lon,np.zeros((5,2))))

Cr = np.array([[-1., 0., 0., 0., 0., 0., 0.],
               [0., 0., 0., 0., -1., 0., 0.]])

Alon_aug = np.vstack((Alonaug_T,Cr))

Blon_aug = np.vstack((B_lon,np.zeros((2,2))))

# Clon_aug = np.array[-1 0 0 0 0 0 0;
#             0 0 0 0 -1 0 0];

umax = 5.
alphamax = 0.5
qmax = 3.5
thetamax = np.pi/4.
hmax = 2.5
u_int_max = 1
h_int_max = 2.5

Qlon = np.array([[1./umax**2., 0., 0., 0., 0., 0., 0.],
                 [0., 1./alphamax**2, 0., 0., 0., 0., 0.],
                 [0., 0., 1./qmax**2., 0., 0., 0., 0.],
                 [0., 0., 0., 1./thetamax**2., 0., 0., 0.],
                 [0., 0., 0., 0., 1./hmax**2., 0., 0.],
                 [0., 0., 0., 0., 0., 1./u_int_max**2., 0.],
                 [0., 0., 0., 0., 0., 0., 1./h_int_max**2.]])


demax = np.radians(0.5)
dtmax = 0.01
Rlon = np.array([[1./demax**2, 0.],[0., 1./dtmax**2]])

Klon_all = lqr(Alon_aug,Blon_aug,Qlon,Rlon);

Klon = Klon_all[:,0:5]
Kilon = Klon_all[:,5:]
# Klon = np.array([[2.3297e-03,   1.0044e-01,  -1.3544e+00,  -2.4719e+01,   1.7060e+00],
#                  [1.0098e+00,  -9.1437e-02,  -4.0277e-04,   2.5987e+00,  -2.5730e-01]])
#
# Kilon = np.array([[-1.6348e-01,  -9.8655e-01],
#                   [-9.8655e-01,   1.6348e-01]])

xlon_eq = np.array([[25., 0., 0., theta_eq, 100.]]).T

ylon_eq = np.array([[25., 100.]]).T

ulon_eq = np.array([[de_eq, dt_eq]]).T

limitlon = np.array([[np.radians(30.), 1.]])

# Klon =
#
#   Columns 1 through 5
#
#    2.3297e-03   1.0044e-01  -1.3544e+00  -2.4719e+01   1.7060e+00
#    1.0098e+00  -9.1437e-02  -4.0277e-04   2.5987e+00  -2.5730e-01
#
#   Columns 6 through 7
#
#    1.6348e-01   9.8655e-01
#    9.8655e-01  -1.6348e-01
