import numpy as np
import sys
from lib.envs.blackjack import BlackjackEnv

def print_observation(observation):
    score, dealer_score, usable_ace = observation
    print('Player score: {} (Usable ace: {}), Dealer score: {}'.format(
        score, usable_ace, dealer_score))

def strategy(observation):
    score, dealer_score, _ = observation
    return 0 if score >= 20 else 1

if __name__ == '__main__':
    env = BlackjackEnv()
    for episode in range(20):
        observation = env.reset()
        while True:
            print_observation(observation)
            action = strategy(observation)
            print('Taking action: {} (0: stick, 1: hit)'.format(action))
            observation, reward, done, _ = env.step(action)
            if done:
                print_observation(observation)
                print('Game end. Reward: {}\n'.format(reward))
                break
