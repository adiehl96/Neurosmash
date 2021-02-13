# Example code that creates plots directly in report
# Code is an implementation of a genetic algorithm

def train(agent, env, episodes):
    agent.epsilon = 0.5
    loss = []
    rewards = []
    for episode in range(episodes):
        agent.choose_goal()
        end, reward, oldstate = env.reset()
        i=0
        while (not end):
            i+=1
            action = agent.step()
            end, reward, newstate = env.step(action)
            if(not end and i>200):
                end = True
            agent.remember(action, reward, newstate, end, train=True)
        rewards.append(reward)
        loss.append(agent.learn(episode))
        print("Training Episode {:3d}, current loss: {:4.6f}, cum reward: {:4.2f}".format(episode, loss[-1], sum(rewards)))
    return rewards

def exploit(agent, env, episodes):
    agent.epsilon = 0.0
    rewards = []
    goals = []
    for episode in range(episodes):
        agent.choose_goal()
        end, _, oldstate = env.reset()
        while (not end):
            action = agent.step()
            end, reward, newstate = env.step(action)
            agent.remember(action, reward, newstate, end)
        if(reward):
            goals.append(newstate)
        rewards.append(reward)
        print("Testing Episode {:3d}, cum reward: {:4.2f}".format(episode, sum(rewards)))
    return rewards, goals