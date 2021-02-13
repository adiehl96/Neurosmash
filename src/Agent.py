import random
from collections import deque

import numpy as np
import mxnet as mx
from mxnet import autograd, gluon, nd

from ReplayBuffer import ReplayBuffer
from QNetwork import QNetwork

class Agent:
    def __init__(self, ctx=mx.cpu(), learning_rate=0.0001, gamma=0.95, epsilon=0.5, stacksize=4, planning=2, targetupdate=10, rewardstuning={}, vae_absent=True, size=64):
        '''set all begin values and hyperparameters of the Agent'''
        self.model = QNetwork(n_actions=3, vae_absent=vae_absent)
        self.model.initialize(mx.init.Xavier(), ctx=ctx)
        self.targetmodel = QNetwork(n_actions=3, vae_absent=vae_absent)
        self.targetmodel.initialize(mx.init.Xavier(), ctx=ctx)
        self.ctx = ctx
        self.stacksize = stacksize
        self.planning = planning
        self.targetupdate = targetupdate
        self.vae_absent = vae_absent
        self.loss = gluon.loss.HuberLoss()
        self.optimizer = gluon.Trainer(self.model.collect_params(), 'adam', {'learning_rate': learning_rate})
        self.gamma = gamma
        self.epsilon = epsilon
        self.hrm = ReplayBuffer()
        self.step_counter = 0
        self.goal = np.empty((0))
        self.oldstates = deque(maxlen=self.stacksize)
        self.newstates = deque(maxlen=self.stacksize)
        self.size = size
        self.rewardstuning = rewardstuning

    def learn(self, episode):
        '''updates the model weights according to memory of the last episode'''
        # Load history from memory
        memory = self.hrm.recall()
        print(len(memory))
        self.max_batch_size = 201
        if(len(memory) < self.max_batch_size): return 0
        self.epoch = self.planning
        
        losslist = []
        for i in range(self.epoch):
            memory = random.sample(memory, len(memory))
            oldstates, actions, rewards, newstates, endstates = zip(*memory)

            oldstates_train_data = mx.io.NDArrayIter(data={'data':np.array(oldstates)}, batch_size = self.max_batch_size, shuffle=False, last_batch_handle='pad')
            actions_train_data = mx.io.NDArrayIter(np.array(actions), batch_size = self.max_batch_size, shuffle=False, last_batch_handle='pad')
            rewards_train_data = mx.io.NDArrayIter(np.array(rewards), batch_size = self.max_batch_size, shuffle=False, last_batch_handle='pad')
            newstates_train_data = mx.io.NDArrayIter(np.array(newstates), batch_size = self.max_batch_size, shuffle=False, last_batch_handle='pad')
            endstates_train_data = mx.io.NDArrayIter(np.array(endstates), batch_size = self.max_batch_size, shuffle=False, last_batch_handle='pad')
            
            train_data = [oldstates_train_data, actions_train_data, rewards_train_data, newstates_train_data, endstates_train_data]
            # Reset the train data iterator.
            [train_data.reset() for train_data in train_data]
            # Loop over the train data iterator.
            for batch in zip(*train_data):
                os_data = gluon.utils.split_and_load(batch[0].data[0], ctx_list=[self.ctx], batch_axis=0)
                a_data = gluon.utils.split_and_load(batch[1].data[0], ctx_list=[self.ctx], batch_axis=0)
                r_data = gluon.utils.split_and_load(batch[2].data[0], ctx_list=[self.ctx], batch_axis=0)
                ns_data = gluon.utils.split_and_load(batch[3].data[0], ctx_list=[self.ctx], batch_axis=0)
                es_data = gluon.utils.split_and_load(batch[4].data[0], ctx_list=[self.ctx], batch_axis=0)
                # Inside training scope
                for os, a, r, ns, es in zip(os_data, a_data, r_data, ns_data, es_data):
                    with autograd.record(train_mode=True):
                        output = self.model(os)
                        # self.model.summary(os)
                    target = output.detach().asnumpy()
                    prediction = self.targetmodel(ns).asnumpy()
                    future_reward = prediction[:,prediction.argmax(axis=1)]
                    target[:, a.detach().asnumpy().astype(np.int8)] = r.detach().asnumpy() + self.gamma * future_reward * (1-es.detach().asnumpy())
                    with autograd.record(train_mode=True):
                        loss = self.loss(output, nd.array(target, ctx=self.ctx)).mean()
                        mx.nd.waitall()
                        autograd.backward(loss)
                        losslist.append(loss)
                    self.optimizer.step(batch[0].data[0].shape[0])
        
        if(episode % self.targetupdate == 0):
            # print("transferring weights")
            if(self.vae_absent):
                self.targetmodel.conv1.weight.data()[:] = self.model.conv1.weight.data()[:]
                self.targetmodel.conv2.weight.data()[:] = self.model.conv2.weight.data()[:]
            self.targetmodel.dense1.weight.data()[:] = self.model.dense1.weight.data()[:]
            self.targetmodel.dense2.weight.data()[:] = self.model.dense2.weight.data()[:]
            self.targetmodel.dense3.weight.data()[:] = self.model.dense3.weight.data()[:]

        return np.mean([loss.mean().asscalar() for loss in losslist])

    def step(self):
        if np.random.rand() < self.epsilon or len(self.oldstates)<self.stacksize:
             # epsilon greedy
            action = np.random.randint(0,3)
        else:
            # Choose action with maximal q value
            inputstate = np.array(list(self.oldstates)+[self.goal]) if (len(self.goal)) else np.array(self.oldstates)
            action = np.argmax(self.model(nd.array(np.expand_dims(inputstate,0), ctx=self.ctx)).asnumpy())

        return action

    def remember(self, action, reward, newstate, end, train=False):
        self.oldstates = self.newstates
        newstate = self.state_processing(newstate)
        self.newstates.append(newstate)
        
        if(len(self.oldstates) == self.stacksize and train):
            if(self.rewardstuning):
                if(not end):
                    reward = self.rewardstuning['constant reward']
                elif(not reward):
                    reward = self.rewardstuning['death']
                else:
                    reward = self.rewardstuning['win']
            
            if(end and reward and len(self.goal)):
                self.goals = np.concatenate((self.goals,[self.newstates[-1]]),axis=0)
            
            oldstate = np.array(list(self.oldstates)+[self.goal]) if (len(self.goal)) else np.array(self.oldstates)
            newstate = np.array(list(self.newstates)+[self.goal]) if (len(self.goal)) else np.array(self.newstates)
            
            self.hrm.remember(oldstate, action, reward, newstate, end)    

    def choose_goal(self):
        self.goal = np.empty((0))

class Random_agent(Agent):
    '''Define the Random Agent, which simply takes random steps'''
    def __init__(self, ctx=mx.cpu()):
        pass

    def step(self):
        return 0
    
    def learn(self, *argv):
        return 0

    def remember(self, *argv):
        pass
    
class Leftist_agent(Agent):
    '''Define the Leftist Agent, which simply returns 1 at all times (indicating to go left)'''
    def __init__(self, ctx=mx.cpu()):
        pass

    def step(self):
        return 1
    
    def learn(self, *argv):
        return 0

    def remember(self, *argv):
        pass

class DQ_Agent(Agent):
    '''Define the standard Deep Q Agent'''
    def __init__(self, ctx=mx.cpu()):
        super(DQ_Agent, self).__init__(ctx = ctx)

    def state_processing(self, state):
        return np.array(state, "uint8").reshape(self.size, self.size, 3)[:,:,0]


class HRM_DQ_Agent(DQ_Agent):
    '''Define the Deep Q Agent adding hindsight experience replay'''
    def __init__(self, ctx=mx.cpu(), goals=[]):
        super(HRM_DQ_Agent, self).__init__(ctx = ctx)
        self.goals = np.asarray([self.state_processing(goal) for goal in goals])

    def choose_goal(self):
        self.goal = self.goals[np.random.choice(self.goals.shape[0])]
    
    def learn(self, episode):
        '''updates the model weights according to memory of the last episode'''
        self.hrm.apply_hindsight()
        return super(HRM_DQ_Agent, self).learn(episode)
