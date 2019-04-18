# dubins_parameters
#   - Dubins parameters that define path between two configurations
#
# mavsim_matlab 
#     - Beard & McLain, PUP, 2012
#     - Update history:  
#         3/26/2019 - RWB

import numpy as np
import sys
sys.path.append('..')


class dubins_parameters:
    def __init__(self):
        self.p_s = np.inf*np.ones((3,1))  # the start position in re^3
        self.chi_s = np.inf  # the start course angle
        self.p_e = np.inf*np.ones((3,1))  # the end position in re^3
        self.chi_e = np.inf  # the end course angle
        self.radius = np.inf  # turn radius
        self.length = np.inf  # length of the Dubins path
        self.center_s = np.inf*np.ones((3,1))  # center of the start circle
        self.dir_s = np.inf  # direction of the start circle
        self.center_e = np.inf*np.ones((3,1))  # center of the end circle
        self.dir_e = np.inf  # direction of the end circle
        self.r1 = np.inf*np.ones((3,1))  # vector in re^3 defining half plane H1
        self.r2 = np.inf*np.ones((3,1))  # vector in re^3 defining position of half plane H2
        self.r3 = np.inf*np.ones((3,1))  # vector in re^3 defining position of half plane H3
        self.n1 = np.inf*np.ones((3,1))  # unit vector in re^3 along straight line path
        self.n3 = np.inf*np.ones((3,1))  # unit vector defining direction of half plane H3

    def update(self, ps, chis, pe, chie, R):
        ell = np.linalg.norm(ps - pe)
        if ell < 2 * R:
            print('Error in Dubins Parameters: The distance between nodes must be larger than 2R.')
        else:
            self.p_s = ps
            self.chi_s = chis
            self.p_e = pe
            self.chi_e = chie
            self.radius = R
            e1 = np.array([[1.,0.,0.]]).T
            crs = ps + R*rotz(np.pi/2.)@np.array([[np.cos(chis),np.sin(chis),0.]]).T
            cls = ps + R*rotz(-np.pi/2.)@np.array([[np.cos(chis),np.sin(chis),0.]]).T
            cre = pe + R*rotz(np.pi/2.)@np.array([[np.cos(chie),np.sin(chie),0.]]).T
            cle = pe + R*rotz(-np.pi/2.)@np.array([[np.cos(chie),np.sin(chie),0.]]).T

            pi = np.pi
            pi_2 = np.pi/2.
            pi2 = np.pi*2.

            vartheta = np.arctan2((cre.item(1) - crs.item(1)),
                                  (cre.item(0) - crs.item(0)))  # at2(diff(e),diff(n))
            L1 = np.linalg.norm(crs-cre)+R*mod(pi2+mod(vartheta-pi_2)-mod(chis-pi_2))+R*mod(pi2+mod(chie-pi_2)-mod(vartheta-pi_2))

            ell = np.linalg.norm(cle-crs)
            vartheta = np.arctan2((cle.item(1) - crs.item(1)),
                                  (cle.item(0) - crs.item(0)))  # at2(diff(e),diff(n))
            vartheta2 = vartheta - pi_2 + np.arcsin(2.*R/ell)
            L2 = np.sqrt(ell**2-4*R**2)+R*mod(pi2+mod(vartheta2)-mod(chis-pi_2))+R*mod(pi2+mod(vartheta2+pi)-mod(chie+pi_2))

            ell = np.linalg.norm(cre-cls)
            vartheta = np.arctan2((cre.item(1) - cls.item(1)),
                                  (cre.item(0) - cls.item(0)))  # at2(diff(e),diff(n))
            vartheta2 = np.arccos(2.*R/ell)
            L3 = np.sqrt(ell**2-4*R**2)+R*mod(pi2+mod(chis+pi_2)-mod(vartheta+vartheta2))+R*mod(pi2+mod(chie-pi_2)-mod(vartheta+vartheta2-pi))

            vartheta = np.arctan2((cle.item(1) - cls.item(1)),
                                  (cle.item(0) - cls.item(0)))  # at2(diff(e),diff(n))
            L4 = np.linalg.norm(cls-cle)+R*mod(pi2+mod(chis+pi_2)-mod(vartheta+pi_2))+R*mod(pi2+mod(vartheta+pi_2)-mod(chie+pi_2))
            L = np.array([L1,L2,L3,L4])
            self.length = np.amin(L)
            minL = np.argmin(L)
            if minL == 0:
                self.center_s = crs
                self.dir_s = 1
                self.center_e = cre
                self.dir_e = 1
                self.n1 = (self.center_e-self.center_s)/np.linalg.norm(self.center_e-self.center_s)#q1
                self.r1 = self.center_s + R*rotz(-np.pi/2.)@self.n1#z1
                self.r2 = self.center_e + R*rotz(-np.pi/2.)@self.n1#z2
            elif minL == 1:
                self.center_s = crs
                self.dir_s = 1
                self.center_e = cle
                self.dir_e = -1
                ell = np.linalg.norm(self.center_e-self.center_s)
                vartheta = np.arctan2((cle.item(1) - crs.item(1)),
                                      (cle.item(0) - crs.item(0)))
                vartheta2 = vartheta - np.pi/2. + np.arcsin(2.*R/ell)
                self.n1 = rotz(vartheta2+np.pi/2.)@e1#q1
                self.r1 = self.center_s + R*rotz(vartheta2)@e1#z1
                self.r2 = self.center_e + R*rotz(vartheta2 + np.pi)@e1#z2
            elif minL == 2:
                self.center_s = cls
                self.dir_s = -1
                self.center_e = cre
                self.dir_e = 1
                ell = np.linalg.norm(self.center_e-self.center_s)
                vartheta = np.arctan2((cre.item(1) - cls.item(1)),
                                      (cre.item(0) - cls.item(0)))
                vartheta2 = np.arccos(2.*R/ell)
                self.n1 = rotz(vartheta+vartheta2-np.pi/2.)@e1#q1
                self.r1 = self.center_s + R*rotz(vartheta+vartheta2)@e1#z1
                self.r2 = self.center_e + R*rotz(vartheta+vartheta2 - np.pi)@e1#z2
            else: # minL == 3
                self.center_s = cls
                self.dir_s = -1
                self.center_e = cle
                self.dir_e = -1
                self.n1 = (self.center_e-self.center_s)/np.linalg.norm(self.center_e-self.center_s)#q1
                self.r1 = self.center_s + R*rotz(np.pi/2.)@self.n1#z1
                self.r2 = self.center_e + R*rotz(np.pi/2.)@self.n1#z2

            self.r3 = pe # z3
            self.n3 = rotz(chie)@e1 #q3


def rotz(theta):
    return np.array([[np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1]])


def mod(x):
    # make x between 0 and 2*pi
    while x < 0:
        x += 2*np.pi
    while x > 2*np.pi:
        x -= 2*np.pi
    return x #% (2*np.pi)


