from random import random

class RandomAgent:

    def choose_action(odds):
        if random() <= odds:
            outcome = 1
        else:
            outcome = 0
        return outcome

test = RandomAgent.choose_action(0.6)
print(test)