betamax = np.pi/2.
pmax = 5
rmax = 1.
phimax = np.pi/2.
psimax = np.pi/2.
beta_int_max = 1
psi_int_max = 1

Qlat = np.array([[1./betamax**2., 0., 0., 0., 0., 0., 0.],
                 [0., 1./pmax**2, 0., 0., 0., 0., 0.],
                 [0., 0., 1./rmax**2., 0., 0., 0., 0.],
                 [0., 0., 0., 1./phimax**2., 0., 0., 0.],
                 [0., 0., 0., 0., 1./psimax**2., 0., 0.],
                 [0., 0., 0., 0., 0., 1./beta_int_max**2., 0.],
                 [0., 0., 0., 0., 0., 0., 1./psi_int_max**2.]])


damax = np.radians(0.1)
drmax = np.radians(0.1)
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

limitlat = np.array([[np.radians(30.), np.radians(30.)]])
