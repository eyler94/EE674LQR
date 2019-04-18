import numpy as np
from math import atan2, asin

def Euler2Quaternion(phi, theta, psi):
    cth = np.cos(theta/2)
    cph = np.cos(phi/2)
    cps = np.cos(psi/2)

    sth = np.sin(theta/2)
    sph = np.sin(phi/2)
    sps = np.sin(psi/2)

    e0 = cps*cth*cph+sps*sth*sph
    e1 = cps*cth*sph-sps*sth*cph
    e2 = cps*sth*cph+sps*cth*sph
    e3 = sps*cth*cph+cps*sth*sph
    norm_e = np.linalg.norm(np.array([e0,e1,e2,e3]))
    e = np.array([e0,e1,e2,e3]/norm_e)
    return e


def Quaternion2Euler(e):
    norm_e = np.linalg.norm(e)
    e0 = e.item(0)/norm_e
    e1 = e.item(1)/norm_e
    e2 = e.item(2)/norm_e
    e3 = e.item(3)/norm_e
    # print("\ne0:\n",e0,"\ne1:\n",e1,"\ne2:\n",e2,"\ne3:\n",e3)

    phi = atan2(2*(e0*e1 + e2*e3),(e0**2+e3**2-e1**2-e2**2))
    theta = asin(2*(e0*e2-e1*e3))
    psi = atan2(2*(e0*e3 + e1*e2),(e0**2+e1**2-e2**2-e3**2))
    return [phi,theta,psi]

def Quaternion2Rotation(e):
    e0 = e.item(0)
    e1 = e.item(1)
    e2 = e.item(2)
    e3 = e.item(3)

    R = np.array([[e0**2 + e1**2 - e2**2 - e3**2, 2*(e1*e2 - e0*e3), 2*(e1*e3 + e0*e2)],
                  [2*(e1*e2 + e0*e3), e0**2 - e1**2 + e2**2 - e3**2, 2*(e2*e3 - e0*e1)],
                  [2*(e1*e3 - e0*e2), 2*(e2*e3 + e0*e1), e0**2 - e1**2 - e2**2 + e3**2]
                  ])
    return R

def Euler2Rotation(phi, theta, psi):
    # print("orig:", phi, theta, psi)
    # e = Euler2Quaternion(phi, theta, psi)
    # R = Quaternion2Euler(e)

    cph = np.cos(phi)
    sph = np.sin(phi)
    cth = np.cos(theta)
    sth = np.sin(theta)
    cps = np.cos(psi)
    sps = np.sin(psi)

    Rbv2 = np.array([[1., 0., 0.],\
                    [0., cph, sph],\
                    [0., -sph, cph]])

    Rv2v1 = np.array([[cth, 0., -sth],\
                    [0., 1., 0.],\
                    [sth, 0., cth]])

    Rv1i = np.array([[cps, sps, 0.],\
                    [-sps, cps, 0.],\
                    [0., 0., 1.]])

    R = Rbv2@Rv2v1@Rv1i

    return R.T