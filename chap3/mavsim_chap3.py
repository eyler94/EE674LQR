"""
mavSimPy 
    - Chapter 3 assignment for Beard & McLain, PUP, 2012
    - Update history:  
        12/18/2018 - RWB
        1/14/2019 - RWB
"""
import sys
sys.path.append('..')
import numpy as np
import parameters.simulation_parameters as SIM
from time import sleep

from chap2.mav_viewer import mav_viewer
# from chap2.video_writer import video_writer
from chap3.data_viewer import data_viewer
from chap3.mav_dynamics import mav_dynamics


# initialize the visualization
VIDEO = False  # True==write video, False==don't write video
mav_view = mav_viewer()  # initialize the mav viewer
data_view = data_viewer()  # initialize view of data plots
if VIDEO == True:
    video = video_writer(video_name="chap3_video.avi",
                         bounding_box=(0, 0, 1000, 1000),
                         output_rate=SIM.ts_video)

# initialize elements of the architecture
mav = mav_dynamics(SIM.ts_simulation)

# initialize the simulation time
sim_time = SIM.start_time

# main simulation loop
print("Press Command-Q to exit...")
# sleep(10)
moment_size = 0.015
while sim_time < SIM.end_time:
    # fx = 0  # 10
    # fy = 0  # 10
    # fz = 0  # 10
    # Mx = 0  # 0.1
    # My = 0  # 0.1
    # Mz = 0.015  # 0.1
    # forces_moments = np.array([[fx, fy, fz, Mx, My, Mz]]).T
    # if sim_time > 12:
    #     fx = 0  # 10
    #     fy = 0  # 10
    #     fz = 0  # 10
    #     Mx = 0  # 0.1
    #     My = 0  # 0.1
    #     Mz = 0  # 0.1
    #     forces_moments = np.array([[fx, fy, fz, Mx, My, Mz]]).T

    if sim_time < SIM.end_time / 50.:
        # -------vary forces and moments to check dynamics-------------
        fx = 10.  # 10
        fy = 0  # 10
        fz = 0  # 10
        Mx = 0  # 0.1
        My = 0  # 0.1
        Mz = 0  # 0.1
        forces_moments = np.array([[fx, fy, fz, Mx, My, Mz]]).T

        # -------physical system-------------
        mav.update_state(forces_moments)  # propagate the MAV dynamics
    elif sim_time <= 2 * SIM.end_time / 50.:
        fx = -10.
        forces_moments = np.array([[fx, fy, fz, Mx, My, Mz]]).T
        mav.update_state(forces_moments)
    elif sim_time <= 3 * SIM.end_time / 50.:
        fx = 0.
        fy = 10.
        forces_moments = np.array([[fx, fy, fz, Mx, My, Mz]]).T
        mav.update_state(forces_moments)
    elif sim_time <= 4 * SIM.end_time / 50.:
        fy = -10.
        forces_moments = np.array([[fx, fy, fz, Mx, My, Mz]]).T
        mav.update_state(forces_moments)
    elif sim_time <= 5 * SIM.end_time / 50.:
        fy = 0.
        fz = -10.
        forces_moments = np.array([[fx, fy, fz, Mx, My, Mz]]).T
        mav.update_state(forces_moments)
    elif sim_time <= 6 * SIM.end_time / 50.:
        fz = 10.
        forces_moments = np.array([[fx, fy, fz, Mx, My, Mz]]).T
        mav.update_state(forces_moments)
    elif sim_time <= 13 * SIM.end_time / 100.:
        fz = 0.
        My = moment_size
        forces_moments = np.array([[fx, fy, fz, Mx, My, Mz]]).T
        mav.update_state(forces_moments)
    elif sim_time <= 15 * SIM.end_time / 100.:
        My = -moment_size
        forces_moments = np.array([[fx, fy, fz, Mx, My, Mz]]).T
        mav.update_state(forces_moments)
    elif sim_time <= 16 * SIM.end_time / 100.:
        My = moment_size
        forces_moments = np.array([[fx, fy, fz, Mx, My, Mz]]).T
        mav.update_state(forces_moments)
    elif sim_time <= 17 * SIM.end_time / 100.:
        My = 0.
        Mx = moment_size
        forces_moments = np.array([[fx, fy, fz, Mx, My, Mz]]).T
        mav.update_state(forces_moments)
    elif sim_time <= 19 * SIM.end_time / 100.:
        Mx = -moment_size
        forces_moments = np.array([[fx, fy, fz, Mx, My, Mz]]).T
        mav.update_state(forces_moments)
    elif sim_time <= 20 * SIM.end_time / 100.:
        Mx = moment_size
        forces_moments = np.array([[fx, fy, fz, Mx, My, Mz]]).T
        mav.update_state(forces_moments)
    elif sim_time <= 21 * SIM.end_time / 100.:
        Mx = 0.
        Mz = moment_size
        forces_moments = np.array([[fx, fy, fz, Mx, My, Mz]]).T
        mav.update_state(forces_moments)
    elif sim_time <= 23 * SIM.end_time / 100.:
        Mz = -moment_size
        forces_moments = np.array([[fx, fy, fz, Mx, My, Mz]]).T
        mav.update_state(forces_moments)
    else:
        Mz = moment_size
        forces_moments = np.array([[fx, fy, fz, Mx, My, Mz]]).T
        mav.update_state(forces_moments)

    # -------physical system-------------
    mav.update_state(forces_moments)  # propagate the MAV dynamics

    # -------update viewer-------------
    mav_view.update(mav.msg_true_state)  # plot body of MAV
    data_view.update(mav.msg_true_state, # true states
                     mav.msg_true_state, # estimated states
                     mav.msg_true_state, # commanded states
                     SIM.ts_simulation)
    if VIDEO == True:
        video.update(sim_time)

    # -------increment time-------------
    sim_time += SIM.ts_simulation

if VIDEO == True:
    video.close()




