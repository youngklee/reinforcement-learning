import numpy as np
from lib.envs.gridworld import GridworldEnv

def value_iteration(env, gamma=1.0, theta=0.0001):
    def one_step_lookahead(s, V):
        action_values = np.zeros(env.nA)
        for a in range(env.nA):
            for prob, ns, reward, _ in env.P[s][a]:
                action_values[a] += prob*(reward + gamma*V[ns])
        return action_values

    V = np.zeros(env.nS)
    while True:
        delta = 0
        for s in range(env.nS):
            action_values = one_step_lookahead(s, V)
            best_action_value = np.max(action_values)
            delta = max(delta, np.abs(best_action_value - V[s]))
            V[s] = best_action_value
        if delta < theta:
            break

    policy = np.zeros([env.nS, env.nA])
    for s in range(env.nS):
        action_values = one_step_lookahead(s, V)
        best_action = np.argmax(action_values)
        policy[s, best_action] = 1.0

    return policy, V

if __name__ == '__main__':
    env = GridworldEnv()
    policy, v = value_iteration(env)
    print('Policy:\n{}'.format(policy))
    print('Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):\n{}'.format(
        np.reshape(np.argmax(policy, axis=1), env.shape)))
    print('Value Function:\n{}'.format(v))
    print('Reshaped Grud Value Function:\n{}'.format(
        v.reshape(env.shape)))
