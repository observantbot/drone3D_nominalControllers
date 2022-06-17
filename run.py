import numpy as np
import time
from PhysicsEngine import EnvPhysicsEngine
import pybullet as p
from pybulletsim import init_simulation, end_simulation
import matplotlib.pyplot as plt

drone, marker = init_simulation(render=True)
pe = EnvPhysicsEngine()

pe.reset(x=0, y=0, z=0,
        phi = np.deg2rad(00), theta = np.deg2rad(00), psi = np.deg2rad(00),
        x_dot = 0, y_dot = 0, z_dot = 0,
        phi_dot = .00, theta_dot = 0.00, psi_dot = 0)

step = 500

def run(step):
    x_log = np.zeros(step)
    y_log = np.zeros(step)
    z_log = np.zeros(step)
    phi_log = np.zeros(step)
    theta_log = np.zeros(step)
    psi_log = np.zeros(step)

    ti = np.zeros(step)
    for i in range(step):

        x_log[i] = pe.curr[0]
        y_log[i] = pe.curr[1]
        z_log[i] = pe.curr[2]
        phi_log[i] = pe.curr[3]
        theta_log[i] = pe.curr[4]
        psi_log[i] = pe.curr[5]


        ti[i] = 0.01*i

        # here you can play around with various desired conditions
        pe.stepSimulation(x_d = 3.0, y_d = 3.0, z_d = 3.0,
                        phi_d=  np.deg2rad(0), theta_d = np.deg2rad(0), psi_d = np.deg2rad(0))
        p.resetBasePositionAndOrientation(drone, pe.curr[:3],
                            p.getQuaternionFromEuler([pe.curr[3], pe.curr[4],pe.curr[5]]))
         
        # set marker to the desired position as well
        p.resetBasePositionAndOrientation(marker, [3, 3.0, 3.0],
                            p.getQuaternionFromEuler([0, 0, 0]))
        
        time.sleep(0.01)

    return ti,x_log,y_log,z_log, phi_log, theta_log, psi_log

ti,x_log,y_log,z_log, phi_log, theta_log, psi_log = run(step)

end_simulation()

