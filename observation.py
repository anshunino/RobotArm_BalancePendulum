from robot_descriptions.loaders.pybullet import load_robot_description
import matplotlib.pyplot as plt
import numpy as np
import pybullet
import time


class Actor():

    def __init__(self):
        self.free_joints = [1, 2, 3, 4, 5]
        self.ee_link_id = 7
        self.INITIAL_CONF = [-0.07195958737978714, -0.031165154579558596, -
                             1.804251465569389, -1.4883759445410973, 0.11663409459107088]

        '''
        episode, ee_positions_x, ee_positions_y, ee_positions_z = self.rollout(
            self.pid_policy, dt=0.01)
        # print('Actions: ', episode['action'], '\n\n')
        # print('Rewards: ', episode['reward'], '\n\n')
        # print('Observations: ', episode['observation'], '\n\n')
        # print('Num steps: ', len(episode['action']))
        plt.plot(np.array(episode['observation'])[:, 0], label='x')
        plt.plot(np.array(episode['observation'])[:, 1], label='theta')
        plt.plot(episode['action'], label='actions')
        plt.legend()
        plt.savefig('response.png')
        plt.close()

        plt.plot(np.array(episode['observation'])[:, 0], label='x')
        plt.plot(np.array(episode['observation'])[:, 1], label='theta')
        plt.plot(np.array(episode['observation'])[:, 2], label='v')
        plt.plot(np.array(episode['observation'])[:, 3], label='w')
        plt.legend()
        plt.savefig('obs.png')
        plt.close()

        plt.plot(episode['action'], label='actions')
        plt.legend()
        plt.savefig('actions.png')
        plt.close()

        plt.plot(ee_positions_x, label='ee_positions_x')
        plt.plot(ee_positions_y, label='ee_positions_y')
        plt.plot(ee_positions_z, label='ee_positions_z')
        plt.legend()
        plt.savefig('EE_positions.png')
        plt.close()'''

    def getObservation(self, robot, link_ids=np.linspace(0, 11, 12, dtype=int), init_pos=np.array([0.36328751177246044, 0.16626388143593868, 0.5289256634238204])):
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

    def getNextObservation(self, x_t, v_t, a_t, dt):
        v_t1 = v_t + a_t*dt
        x_t1 = x_t + v_t*dt + 0.5*a_t*dt*dt
        return (x_t1, v_t1)

    def getNextStep(self, robot, observation_t, policy, dt, replay_buffer):
        x_t, theta_t, v_t, w_t = observation_t
        a_t = policy(observation_t, replay_buffer, dt)

        dx = v_t*dt + 0.5*a_t*dt*dt

        p, theta_t1, _ = self.getObservation(robot)
        w_t1 = (theta_t1 - theta_t)/dt

        x_t1, v_t1 = self.getNextObservation(x_t, v_t, a_t, dt)

        pos, quat = self.get_ee_pose(robot)

        tgt_jnt_poss = self.compute_inverse_kinematics(
            robot, np.array(pos)+dx*np.array([1, 0, 0]).flatten().tolist(), quat)
        for i in range(4):
            self.set_jpos(tgt_jnt_poss, robot, dt/4)
            # pybullet.setJointMotorControlArray(
            #    robot, self.free_joints, pybullet.POSITION_CONTROL, targetPositions=tgt_jnt_poss)
            # pybullet.stepSimulation()
        pybullet.setTimeStep(dt)
        time.sleep(dt)

        observation_t1 = [x_t1[0], theta_t1, v_t1[0], w_t1]

        terminated = self.getTerminationState(observation_t1)
        reward = self.getReward(observation_t1, terminated)

        return a_t, observation_t1, reward, terminated

    def getReward(self, observation, terminated):
        if terminated:
            reward = 0.0
        else:
            reward = 1.0

        return reward

    def getTerminationState(self, observation):
        x, theta, _, _ = observation
        terminated = bool(
            x < -2.4
            or x > 2.4
            or theta < -0.20944
            or theta > 0.20944
        )
        return terminated

    def pid_policy(self, observation, replay_buffer, dt):
        '''
        Calculates the PID control to determine the action for the next step
        Args:
        --- observation: Array containing x_t (position), theta_t (vertical angle), v_t (velocity) and angular velocity
        Ret:
        --- a: action vector: desired acceleration for next step
        '''
        o = np.array([0., 0., 0., 0.])
        # d_t = 0.02

        K_p = 500*np.array([0.4, 8., 0.01, 0.005])
        K_d = 10.*np.sqrt(K_p)  # 0.1*np.array([1., 10., .000001, 1e-7])
        K_i = 0.0002*np.array([0.1, 0.5, 0, 0])
        e_k = np.array(observation) - o
        if len(replay_buffer) > 0:
            d_e_k = (observation - replay_buffer[-1])/dt
        else:
            d_e_k = np.zeros(4)
        i_e_k = np.sum(np.array(replay_buffer), axis=0) * dt
        # print(e_k, d_e_k)

        # print(K_d, d_e_k, np.dot(K_d, d_e_k))
        # + np.dot(K_d, d_e_k) + np.dot(K_i, i_e_k)   # u = -Kx
        a = -1.*np.dot(K_p, e_k) - np.dot(K_d, d_e_k) - np.dot(K_i, i_e_k)

        def sigmoid(x):
            return 1.0 / (1.0 + np.exp(-x))

        # action = sigmoid(u)
        # action = np.round(action).astype(np.int32)
        # print(e_k, u, action)

        return np.array([a])

    def sin_policy(self, observation, replay_buffer, dt):
        o = np.array([0., 0., 0., 0.])
        e_k = observation - o
        # -50.*e_k[0]  # **2-80.*observation[1]
        a = -150*np.tan(e_k[1]) - 50.*e_k[1] - 5.*e_k[0]
        return np.array([a])

    def rollout_from_env(self, robot, policy, dt, replay_size=4):
        episode = {'action': [], 'reward': [], 'observation': []}
        observation = [0.0, self.getObservation(robot)[1], 0.0, 0.0]

        ee_positions_x = []
        ee_positions_y = []
        ee_positions_z = []

        episode['observation'].append(observation)
        for step in range(10000):
            if len(episode['action']) < replay_size:
                replay_buffer = np.array(episode['observation'])
            else:
                replay_buffer = np.array(episode['observation'][-replay_size:])

            action, observation, reward, terminated = self.getNextStep(
                robot, observation, policy, dt, replay_buffer)  # robot.step(action)
            episode['action'].append(action[0])
            episode['reward'].append(reward)
            episode['observation'].append(observation)

            ee_positions_x.append(pybullet.getLinkState(
                robot, self.ee_link_id)[0][0])
            ee_positions_y.append(pybullet.getLinkState(
                robot, self.ee_link_id)[0][1])
            ee_positions_z.append(pybullet.getLinkState(
                robot, self.ee_link_id)[0][2])
            if terminated:
                return episode, ee_positions_x, ee_positions_y, ee_positions_z
        return episode, ee_positions_x, ee_positions_y, ee_positions_z

    def rollout(self, policy, dt: float = 0.01):
        pybullet.connect(pybullet.GUI)
        name = 'ur5_description'
        robot = load_robot_description(name)

        timeStep = 1./300
        pybullet.setTimeStep(timeStep)

        pybullet.setGravity(0, 0, -9.8)

        for j in self.free_joints:  # range (pybullet.getNumJoints(robot)):
            pybullet.setJointMotorControl2(
                robot, j, pybullet.VELOCITY_CONTROL, force=0)
        pybullet.setJointMotorControl2(
            robot, 6, pybullet.VELOCITY_CONTROL, force=0)

        pos, quat = self.get_ee_pose(robot)

        for i in range(100):
            self.set_jpos(self.INITIAL_CONF, robot, timeStep)

        timeStep = 1./3000
        pybullet.setTimeStep(timeStep)
        episode, ee_positions_x, ee_positions_y, ee_positions_z = self.rollout_from_env(
            robot, policy, timeStep)

        pybullet.disconnect()
        return episode, ee_positions_x, ee_positions_y, ee_positions_z

    def compute_inverse_kinematics(self, robot, pos, ori):
        ori = self.to_quat(ori)
        jnt_poss = pybullet.calculateInverseKinematics(
            robot, self.ee_link_id, pos, ori)
        jnt_poss = list(map(self.ang_in_mpi_ppi, jnt_poss))
        arm_jnt_poss = [jnt_poss[i] for i in range(len(self.free_joints))]
        return arm_jnt_poss

    def ang_in_mpi_ppi(self, angle):
        """
        Restricts the angle within the range [-pi, pi)
        """
        angle = (angle + np.pi) % (2 * np.pi) - np.pi
        return angle

    def get_ee_pose(self, robot):
        info = pybullet.getLinkState(robot, self.ee_link_id)
        pos = info[4]
        quat = info[5]
        return np.array(pos), np.array(quat)

    def to_quat(self, ori):
        """
        Converts the input rotation format to unit quaternion
        """
        ori = np.array(ori)
        return ori

    def set_jpos(self, position, robot, timeStep):
        pybullet.setTimeStep(timeStep)
        position = position.copy()
        tgt_pos = position
        pybullet.setJointMotorControlArray(
            robot, self.free_joints, pybullet.POSITION_CONTROL, targetPositions=tgt_pos)
        # pybullet.applyExternalForce(robot, 2, [20, 0, 0], [0.1,0,0], pybullet.LINK_FRAME) #/np.linalg.norm(position)
        pybullet.stepSimulation()
        time.sleep(timeStep)


actor = Actor()
episode, ee_positions_x, ee_positions_y, ee_positions_z = actor.rollout(
    actor.sin_policy, dt=0.01)

plt.plot(np.array(episode['observation'])[:, 0], label='Position')
plt.plot(np.array(episode['observation'])[:, 1], label='Vertical Angle')
plt.legend()
plt.xlabel('Steps')
plt.savefig('obs_order1.png')
plt.close()

plt.plot(np.array(episode['observation'])[:, 0], label='Position')
plt.plot(np.array(episode['observation'])[:, 1], label='Vertical Angle')
plt.plot(np.array(episode['observation'])[:, 2], label='Velocity')
plt.plot(np.array(episode['observation'])[:, 3], label='Angular Velocity')
plt.xlabel('Steps')
plt.legend()
plt.savefig('obs.png')
plt.close()

plt.plot(episode['action'], label='actions')
plt.xlabel('Steps')
plt.legend()
plt.savefig('actions.png')
plt.close()

plt.plot(ee_positions_x, label='ee_positions_x')
plt.plot(ee_positions_y, label='ee_positions_y')
plt.plot(ee_positions_z, label='ee_positions_z')
plt.xlabel('Steps')
plt.legend()
plt.savefig('EE_positions.png')
plt.close()
