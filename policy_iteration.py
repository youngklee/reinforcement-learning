import numpy as np
from policy_eval import policy_eval
from lib.envs.gridworld import GridworldEnv

def policy_improvement(env, policy_eval_fn=policy_eval, gamma=1.0):
    # Start with a random policy
    policy = np.ones([env.nS, env.nA])/env.nA
    while True:
        V = policy_eval_fn(policy, env, gamma)
        stable = True
        for s in range(env.nS):
            best_action = np.argmax(policy[s])
            action_values = np.zeros(env.nA)
            for a in range(env.nA):
                for prob, ns, reward, _ in env.P[s][a]:
                    action_values[a] += prob*(reward + gamma*V[ns])
                new_best_action = np.argmax(action_values)
            if not best_action == new_best_action:
                stable = False
            policy[s] = np.eye(env.nA)[new_best_action]

        if stable:
            return policy, V

if __name__ == '__main__':
    env = GridworldEnv()
    policy, v = policy_improvement(env)
    print('Policy:\n{}'.format(policy))
    print('Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):\n{}'.format(
        np.reshape(np.argmax(policy, axis=1), env.shape)))
    print('Value Function:\n{}'.format(v))
    print('Reshaped Grud Value Function:\n{}'.format(
        v.reshape(env.shape)))
