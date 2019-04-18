"""
compute_trim 
    - Chapter 5 assignment for Beard & McLain, PUP, 2012
    - Update history:  
        2/5/2019 - RWB
"""
import sys
sys.path.append('..')
import numpy as np
from scipy.optimize import minimize
from tools.tools import Euler2Quaternion


def compute_trim(mav, Va, gamma):
    # define initial state and input
    e = Euler2Quaternion(0., gamma, 0.)
    state0 = np.array([[0],  # (0)
                       [0],   # (1)
                       [mav._state[2]],   # (2)
                       [Va],    # (3)
                       [0],    # (4)
                       [0],    # (5)
                       [e.item(0)],    # (6)
                       [e.item(1)],    # (7)
                       [e.item(2)],    # (8)
                       [e.item(3)],    # (9)
                       [0],    # (10)
                       [0],    # (11)
                       [0]     # (12)
                       ])
    delta0 = np.array([
                       [0.],  #de
                       [0.5], #dt
                       [0.0], #da
                       [0.0], #dr
                       ])
    x0 = np.concatenate((state0, delta0), axis=0)
    # define equality constraints
    bnds = ((None, None),(None, None),(None, None),(None, None),\
            (None, None),(None, None),(None, None),(None, None),\
            (None, None),(None, None),(None, None),(None, None),(None, None),\
            (-1.0,1.0),(-1.0,1.0),(-1.0,1.0),(-1.0,1.0))
    cons = ({'type': 'eq',
             'fun': lambda x: np.array([
                                x[3]**2 + x[4]**2 + x[5]**2 - Va**2,  # magnitude of velocity vector is Va
                                x[4],  # v=0, force side velocity to be zero
                                x[6]**2 + x[7]**2 + x[8]**2 + x[9]**2 - 1.,  # force quaternion to be unit length
                                x[7], # e1=0  - forcing e1=e3=0 ensures zero roll and zero yaw in trim
                                x[9], # e3=0
                                x[10], # p=0  - angular rates should all be zero
                                x[11], # q=0
                                x[12], # r=0
                                ]),
             'jac': lambda x: np.array([
                                [0., 0., 0., 2*x[3], 2*x[4], 2*x[5], 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 2*x[6], 2*x[7], 2*x[8], 2*x[9], 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
                                ])
             })
    # solve the minimization problem to find the trim states and inputs
    res = minimize(trim_objective, x0, method='SLSQP', args = (mav, Va, gamma),bounds=bnds,
                   constraints=cons, options={'ftol': 1e-10, 'disp': True})
    # extract trim state and input and return
    trim_state = np.array([res.x[0:13]]).T
    trim_input = np.array([res.x[13:17]]).T
    return trim_state, trim_input

# objective function to be minimized
def trim_objective(x, mav, Va, gamma):
    state = x[0:13].reshape(13,1)
    delta = x[13:17].reshape(4,1)
    xdot = np.array([[0],[0],[-Va*np.sin(gamma)],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0]])
    mav._state = state
    mav._update_velocity_data()
    forces_moments = mav._forces_moments(delta)

    # print("\nfm:", forces_moments)

    f = mav._derivatives(state, forces_moments)
    temp_function = xdot-f
    J = np.linalg.norm(temp_function[2:13])**2.0
    return J

