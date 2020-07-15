import gym

import matplotlib.pyplot as plt
from PIL import Image
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as T
import sys
import random

device = 'cuda'
discount_factor = 0.99


class DQN(nn.Module):

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


def screen(env):
    img = env.render(mode='rgb_array').transpose(2,0,1)
    img = np.ascontiguousarray(img, dtype=np.float32) / 255
    img = torch.from_numpy(img)
    
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize(84),
        T.ToTensor()
    ])

    return transform(img).unsqueeze(0).to(device)


def select_action(dqn, img, threshold):

    if random.random() < threshold:
        with torch.no_grad():
            output = dqn(img)
            
        return output.max(1)[1].item()
    else:
        return random.randint(0,1)


def train(dqn, target, optimizer, env_memory, env_next_memory, action_memory, reward_memory):
    
    batch_size = 128
    criterion = nn.MSELoss()

    dqn.train()

    memory = list(zip(env_memory, env_next_memory, action_memory,reward_memory))
    memory_sample = random.sample(memory, batch_size)

    env_sampled = [x[0] for x in memory_sample]
    env_sampled = torch.cat(env_sampled)
    
    env_next_sampled = [x[1] for x in memory_sample]
    env_next_sampled = torch.cat(env_next_sampled)

    action_sampled = [x[2] for x in memory_sample]

    Q = dqn(env_sampled.to(device))
    Q = Q[torch.arange(Q.size(0)), action_sampled]
    Q_next = target(env_next_sampled.to(device))
    
    Q_target = torch.zeros(batch_size).to(device)
    for i in range(len(memory_sample)):
        if memory_sample[i][3] == -1:
            Q_target[i] = -1
        else:
            Q_target[i] = memory_sample[i][3] + discount_factor * Q_next[i].max(0)[0]

    optimizer.zero_grad()
    loss = criterion(Q_target, Q)
    loss.backward()
    optimizer.step()
    dqn.eval()

    return loss.item()
    

def plot_screen(img):

    # for test
    
    plt.figure()
    plt.imshow(img.squeeze(0).cpu().permute(1,2,0))
    plt.show()

if __name__ == "__main__":
    
    import time

    total_episode = 300
    done = False
    threshold = 0.9
    duration = 0
    duration_list = []

    env = gym.make('CartPole-v0')
    env.reset()

    env_memory = deque(maxlen=10000)
    env_next_memory = deque(maxlen=10000)
    action_memory = deque(maxlen=10000)
    reward_memory = deque(maxlen=10000)

    init_screen = screen(env)
    _, _, screen_height, screen_width = init_screen.shape

    dqn = DQN(screen_height, screen_width, 2)
    target = DQN(screen_height, screen_width, 2)
    target.load_state_dict(dqn.state_dict())
    dqn.to(device)
    target.to(device)


    optimizer = optim.RMSprop(dqn.parameters(), lr=0.001)

    for i in range(total_episode):

        while done != True:
            
            duration += 1
            img = screen(env)
            
            #plot_screen(img)
            env_memory.appendleft(img)
            action = select_action(dqn, img, threshold)
            action_memory.appendleft(action)
            _, reward, done, info = env.step(action)
            
            img_next = screen(env)
            env_next_memory.appendleft(img_next)

            if len(reward_memory) > 256:
                loss = train(dqn, target, optimizer, env_memory, env_next_memory, action_memory, reward_memory)

            if done == True:
                reward_memory.appendleft(-1)
                env.reset()
                duration_list.append(duration)
                duration = 0

            else:
                reward_memory.appendleft(reward)

            env.render()
        
        if len(reward_memory) > 256 and i % 5 == 0:
            target.load_state_dict(dqn.state_dict())
            threshold = (total_episode-i)/ total_episode * 0.9

        env.reset()
        done = False
        print('{} th episode : {}'.format(i+1, duration_list[-1]))
        

    duration_avg = []
    for i in range(len(duration_list)//10):
        duration_avg.append(sum(duration_list[i*10:i*10+10])//10)

    plt.figure()
    plt.plot(duration_avg)
    plt.show()
    env.close()

    import pickle

    with open('duration', 'wb') as f:
        pickle.dump(duration_avg, f)
