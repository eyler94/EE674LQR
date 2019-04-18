"""
mavSimPy 
    - Chapter 2 assignment for Beard & McLain, PUP, 2012
    - Update history:  
        1/10/2019 - RWB
"""
import sys
sys.path.append('..')
import numpy as np
import parameters.simulation_parameters as SIM

# load message types
from message_types.msg_state import msg_state
state = msg_state()  # instantiate state message

#from chap2.mav_viewer import mav_viewer
from chap2.mav_viewer import mav_viewer

# initialize the mav viewer
#mav_view = mav_viewer()
mav_view = mav_viewer()


# initialize the simulation time
sim_time = SIM.start_time

# main simulation loop
while sim_time < SIM.end_time:
    #-------vary states to check viewer-------------
    T = 10
    if sim_time < T:
        state.pn += 10*SIM.ts_simulation
    elif sim_time < 5*T:
        state.pe += 10*SIM.ts_simulation
    elif sim_time < 10*T:
        state.h += 10*SIM.ts_simulation
    elif sim_time < 15*T:
        state.phi += 0.01*SIM.ts_simulation
    elif sim_time < 20*T:
        state.theta += 0.01*SIM.ts_simulation
    else:
        state.psi += 0.1*SIM.ts_simulation

    #-------update viewer-------------
    mav_view.update(state)

    #-------increment time-------------
    sim_time += SIM.ts_simulation

print("Press Ctrl-Q to exit...")
pg.QtGui.QApplication.instance().exec_()




