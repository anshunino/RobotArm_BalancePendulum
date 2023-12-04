from robot_descriptions.loaders.pybullet import load_robot_description
import numpy as np
import pybullet


def getObservation(robot, link_ids, init_pos=np.array([0.36328751177246044, 0.16626388143593868, 0.5289256634238204])):
    '''
    Calculates observation list for current timestep
    Args:
    --- robot: Pybullet object of robot
    Ret:
    --- p: Signed magnitiude of position vector of end-effector with respect to the original point
    --- theta: Vertical angle of pendulum tip with respect to the end-effector
    --- v: Signed magnitiude of end-effector velocity with respect to the original point
    --- omega: Signed magnitiude of pendulum's angular velocity with respect to the end-effector
    '''
    link_states = pybullet.getLinkStates(robot, link_ids)
    world_frame_pos = np.array(
        link_states[9][0]) - np.array(link_states[8][0])

    world_frame_pos *= np.array([1., 0., 1.])

    p = np.array(link_states[8][0]) - init_pos
    theta = np.arccos(np.dot(world_frame_pos, np.array(
        [0., 0., 1.])) / np.linalg.norm(world_frame_pos))

    if world_frame_pos[0] > 0.:
        theta = -theta

    return p, theta, world_frame_pos
