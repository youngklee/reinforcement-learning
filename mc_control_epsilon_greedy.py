import sys
import numpy as np
from collections import defaultdict
import gym
import matplotlib
from lib.envs.blackjack import BlackjackEnv
from lib import plotting

def make_epsilon_greedy_policy(Q, epsilon, num_actions):
    # Create a function that accepts state
    def policy_fn(state):
        A = np.ones(num_actions, dtype=float)*epsilon/num_actions
        best_action = np.argmax(Q[state])
        A[best_action] += (1.0 - epsilon)
        return A
    return policy_fn

def mc_control_epsilon_greedy(env, num_episodes, gamma=1.0, epsilon=0.1):
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)
    for i_episode in range(1, num_episodes):
        if i_episode % 1000 == 0:
            print('\rEpisode {}/{}'.format(i_episode, num_episodes), end='')
            sys.stdout.flush()

        episode = []
        state = env.reset()
        while True:
            probs = policy(state)
            action = np.random.choice(np.arange(len(probs)), p=probs)
            ns, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            if done:
                break
            state = ns

        sa_in_episode = set([(tuple(state), action) for state, action, _ in episode])
        for s, a in sa_in_episode:
            sa = (s, a)
            first_occurence_idx = next(i for i, (state, action, _) in enumerate(episode) if state == s and action == a)
            G = sum([gamma**i*reward for i, (_, _, reward) in enumerate(episode[first_occurence_idx:])])
            returns_sum[sa] += G
            returns_count[sa] += 1
            Q[s][a] = returns_sum[sa]/returns_count[sa]

    return Q, policy

if __name__ == '__main__':
    env = BlackjackEnv()
    Q, policy = mc_control_epsilon_greedy(env, 500000, 0.1)
    V = defaultdict(float)
    for state, action_values in Q.items():
        action_value = np.max(action_values)
        V[state] = action_value
    plotting.plot_value_function(V, title='Optimal Value Function')
