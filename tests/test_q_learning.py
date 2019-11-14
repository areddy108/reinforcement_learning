import gym
import numpy as np
import matplotlib.pyplot as plt


def test_q_learning_slots():
    """
    Tests that the Qlearning implementation successfully finds the slot
    machine with the largest expected reward.
    """
    from code import QLearning

    np.random.seed(0)

    env = gym.make('SlotMachines-v0', n_machines=10, mean_range=(-10, 10), std_range=(5, 10))
    env.seed(0)
    means = np.array([m.mean for m in env.machines])

    agent = QLearning(epsilon=0.2, discount=0)
    state_action_values, rewards = agent.fit(env, steps=10000)

    assert state_action_values.shape == (1, 10)
    assert len(rewards) == 100
    assert np.argmax(means) == np.argmax(state_action_values)

    states, actions, rewards = agent.predict(env, state_action_values)
    assert len(actions) == 1 and actions[0] == np.argmax(means)
    assert len(states) == 1 and states[0] == 0
    assert len(rewards) == 1

def test_q_learning_frozen_lake():
    """
    Tests that the QLearning implementation successfully learns the
    FrozenLake-v0 environment.
    """
    from code import QLearning

    np.random.seed(0)

    env = gym.make('FrozenLake-v0')
    env.seed(0)

    agent = QLearning(epsilon=0.2, discount=0.95)
    state_action_values, rewards = agent.fit(env, steps=10000)

    state_values = np.max(state_action_values, axis=1)

    assert state_action_values.shape == (16, 4)
    assert len(rewards) == 100

    assert np.allclose(state_values[np.array([5, 7, 11, 12, 15])], np.zeros(5))
    assert np.all(state_values[np.array([0, 1, 2, 3, 4, 6, 8, 9, 10, 13, 14])] > 0)


def test_q_learning_deterministic():
    """
    Tests that the QLearning implementation successfully navigates a
    deterministic environment with provided state-action-values.
    """
    from code import QLearning

    np.random.seed(0)

    env = gym.make('FrozonLakeNoSlippery-v0')
    env.seed(0)

    agent = QLearning(epsilon=0.5, discount=0.95)
    state_action_values = np.array([
        [0.0, 0.7, 0.3, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.51, 0.49, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.5, 0.0, 0.5, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.2, 0.8, 0.0],
        [0.0, 0.2, 0.8, 0.0],
        [0.0, 0.6, 0.4, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0]
    ])

    states, actions, rewards = agent.predict(env, state_action_values)
    assert np.all(states == np.array([4, 8, 9, 10, 14, 15]))
    assert np.all(actions == np.array([1, 1, 2, 2, 1, 2]))
    assert np.all(rewards == np.array([0, 0, 0, 0, 0, 1]))

    '''
    env = gym.make('FrozenLake-v0')

    rewards1 = np.zeros([10, 100])
    rewards2 = np.zeros([10, 100])
    rewards3 = np.zeros([10, 100])

    rewards4 = 0;
    rewards5 = 0;
    rewards6 = 0;
    for i in range(10):
        agent1 = QLearning(epsilon=.01, discount =.95, adaptive=False)
        agent2 = QLearning(epsilon=.5, discount = .95)
        agent3 = QLearning(epsilon=.5, discount=.95, adaptive=True)
        state_action_values, rewards = agent1.fit(env, steps=100000)
        states, actions, values = agent1.predict(env, state_action_values)
        rewards1[i, ] = rewards
        rewards4 += np.sum(values)

        print('Abhi')
        state_action_values, rewards = agent2.fit(env, steps=100000)
        states, actions, values = agent2.predict(env, state_action_values)
        rewards2[i,] = rewards
        rewards5 += np.sum(values)

        print('Korand')
        state_action_values, rewards = agent3.fit(env, steps=100000)
        states, actions, values = agent3.predict(env, state_action_values)
        rewards3[i,] = rewards
        rewards6 += np.sum(values)

        print('Red')

    print(rewards4, rewards5, rewards6)
    line1, = plt.plot(np.arange(100), np.mean(rewards1[0:9, :], axis = 0), label='Epsilon = .01, adaptive = False')

    line2, = plt.plot(np.arange(100), np.mean(rewards2[0:9, :], axis=0), label='Epsilon = .5, adaptive = False')

    line3, = plt.plot(np.arange(100), np.mean(rewards3[0:9, :], axis=0), label='Epsilon = .5, adaptive = True')
    plt.legend()
    plt.ylabel('Rewards')
    plt.xlabel('1000 Steps')
    plt.title('QLearning on Frozen Lake with different Parameters (epsilon, adaptive)')
    plt.show()
    
    from code import MultiArmedBandit
    
    env1 = gym.make('SlotMachines-v0')
    env2 = gym.make('FrozenLake-v0')

    rewardsSlotM = np.zeros([10, 100])
    rewardsFrozenM = np.zeros([10, 100])
    rewardsSlotQ = np.zeros([10, 100])
    rewardsFrozenQ = np.zeros([10, 100])

    print('reddy')
    for i in range(10):
        agent = MultiArmedBandit(epsilon=0.2)
        q = QLearning(epsilon=.2, discount=.95, adaptive=False)
        print('1')
        state_action_values, rewards = agent.fit(env1, steps=100000)
        rewardsSlotM[i,] = rewards
        # print(rewards)
        print('2')
        state_action_values, rewards = q.fit(env1, steps=100000)
        rewardsSlotQ[i,] = rewards
        print('3')
        state_action_values, rewards = agent.fit(env2, steps=100000)
        rewardsFrozenM[i,] = rewards
        print('4')
        state_action_values, rewards = q.fit(env2, steps=100000)
        rewardsFrozenQ[i,] = rewards
        print('5')



    line, = plt.plot(np.arange(100), rewardsSlotM[0, :], label='1st trial, Slot-Machines, Bandit')
    line, = plt.plot(np.arange(100), np.mean(rewardsSlotM[0:4, ], axis = 0), label='average 5 trials, Slot-Machines, Bandit')
    line,  =plt.plot(np.arange(100), np.mean(rewardsSlotM[0:9, ], axis = 0), label='average 10 trials, Slot-Machines, Bandit' )
    plt.legend()
    plt.ylabel('Rewards')
    plt.xlabel('1000 Steps')
    plt.title('Multi-Armed Bandit on Slot-Machines')
    plt.show()


    line, = plt.plot(np.arange(100), rewardsSlotM[0, :], label='1st trial, Slot-Machines, Bandit')
    line, = plt.plot(np.arange(100), np.mean(rewardsSlotM[0:4, ], axis=0),
                     label='average 5 trials, Slot-Machines, Bandit')
    line, = plt.plot(np.arange(100), np.mean(rewardsSlotM[0:9, ], axis=0),
                     label='average 10 trials, Slot-Machines, Bandit')
    line, =  plt.plot(np.arange(100), rewardsSlotQ[0, :], label='1st trial, Slot-Machines, Q-Learning')
    line, =  plt.plot(np.arange(100), np.mean(rewardsSlotQ[0:4, ], axis = 0), label='average 5 trials, Slot-Machines, Q-Learning')
    line, =  plt.plot(np.arange(100), np.mean(rewardsSlotQ[0:9, ], axis = 0), label='average 10 trials, Slot-Machines, Q-Learning')
    plt.legend()
    plt.ylabel('Rewards')
    plt.xlabel('1000 Steps')
    plt.title('Multi-Armed Bandit And Q-Learning on Slot-Machines')
    plt.show()



    line, =  plt.plot(np.arange(100), rewardsFrozenM[0, :], label='1st trial, Frozen-Lake, Bandit')
    line, = plt.plot(np.arange(100), np.mean(rewardsFrozenM[0:4, ], axis = 0), label='average 5 trials, Frozen-Lake, Bandit')
    line, = plt.plot(np.arange(100), np.mean(rewardsFrozenM[0:9, ], axis = 0), label='average 10 trials, Frozen-Lake, Bandit')
    line, = plt.plot(np.arange(100), rewardsFrozenQ[0, :], label='1st trial, Frozen-Lake, Q-Learning')
    line, = plt.plot(np.arange(100), np.mean(rewardsFrozenQ[0:4, ], axis = 0), label='average 5 trials, Frozen-Lake, Q-Learning')
    line, = plt.plot(np.arange(100), np.mean(rewardsFrozenQ[0:9, ], axis = 0), label='average 10 trials, Frozen-Lake, Q-Learning')
    plt.legend()
    plt.ylabel('Rewards')
    plt.xlabel('1000 Steps')
    plt.title('Mult-Armed Bandit and QLearning on Frozen Lake')
    plt.show()
    
    '''

