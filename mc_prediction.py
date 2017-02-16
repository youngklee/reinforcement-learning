import numpy as np
from collections import defaultdict
from lib.envs.blackjack import BlackjackEnv
from lib import plotting

def mc_prediction(policy, env, num_episodes, gamma=1.0):
#
# policy is a function that accepts state and returns action probabilities.
#
    V = defaultdict(float)
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)

    for i_episode in range(1, num_episodes + 1):
        # episode is a list of (state, action, reward).
        episode = []
        # state is (score, dealer_score, usable_ace).
        state = env.reset()
        while True:
            probs = policy(state)
            action = np.random.choice(np.arange(len(probs)), p=probs)
            ns, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            if done:
                break
            state = ns

        states_in_episode = set([tuple(state) for state, _, _ in episode])
        for s in states_in_episode:
            first_occurence_idx = next(i for i, (state, _, _) in enumerate(episode) if s == state)
            G = sum([gamma**i*reward for i, (_, _, reward) in enumerate(episode[first_occurence_idx:])])

            # sum over sample episodes
            returns_sum[s] += G
            returns_count[s] += 1.0
            V[s] = returns_sum[s]/returns_count[s]
    return V

def sample_policy(state):
    score, _, _ = state
    return np.array([1.0, 0.0]) if score >= 20 else np.array([0.0, 1.0])

if __name__ == '__main__':
    env = BlackjackEnv()
    V = mc_prediction(sample_policy, env, 10000)
    plotting.plot_value_function(V, title='10,000 Steps')
