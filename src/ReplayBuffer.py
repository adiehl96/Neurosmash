import numpy as np
from collections import deque

class ReplayBuffer:
    def __init__(self):
        self.buffer = deque(maxlen=10000)
        self.episodebuffer = deque()

    def remember(self, oldstate, action, reward, newstate, end):
        self.episodebuffer.append([np.array(oldstate), action, reward, np.array(newstate), end])
        self.buffer.append([np.array(oldstate), action, reward, np.array(newstate), end])

    def apply_hindsight(self):
        height = self.episodebuffer[-1][3].shape[1]
        extra_goal = self.episodebuffer[-1][3][-1]
        for transition in self.episodebuffer:
            transition[0][-1] = extra_goal
            transition[3][-1] = extra_goal
            if (np.all(transition[3][-2] == extra_goal)):
                transition[2] = 1
                transition[4] = 1
            self.buffer.append(transition)
        self.episodebuffer = deque()

    def recall(self):
        return self.buffer