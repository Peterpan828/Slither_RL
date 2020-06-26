import numpy as np
import pyautogui
import imutils
import cv2
import sys
import time
import random
import math
import pytesseract

import mss
import mss.tools
from PIL import Image
import PIL.ImageOps
import re
import os
from selenium import webdriver
from Xlib import display, X

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from collections import namedtuple
from argument import parser

device = 'cuda:0'
discount_factor = 0.98


def open_and_size_browser_window(width, height, x_pos=0, y_pos=0, url='http://www.slither.io'):

    # opens the browser window
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument("--disable-infobars")
    driver = webdriver.Chrome("./chromedriver", chrome_options=chrome_options)
    driver.set_window_size(width, height)

    driver.set_window_position(x_pos, y_pos)
    driver.get(url)

    return driver


class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return x


class DQN(nn.Module):

    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 10, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(10)
        self.conv3 = nn.Conv2d(10, 16, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(16)
        self.conv4 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn4 = nn.BatchNorm2d(32)
        self.flatten = Flatten()
        self.dense1 = nn.Linear(3584, 1024)
        self.dense2 = nn.Linear(1024, 256)
        self.dense3 = nn.Linear(256, 8)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.flatten(x)

        x = F.relu(self.dense1(x))
        x = F.relu(self.dense2(x))
        x = self.dense3(x)

        return x


def action(number, click):
    radian = 2 * math.pi * number / 8
    move_to_radians(radian, click=click)


def move_to_radians(radians, click, radius=100):

    if click == 0:
        pyautogui.moveTo(728 + radius * math.cos(radians),
                         492 + radius * math.sin(radians))
    else:
        pyautogui.mouseDown(728 + radius * math.cos(radians),
                            492 + radius * math.sin(radians))
        time.sleep(0.1)
        pyautogui.mouseUp(728 + radius * math.cos(radians),
                          492 + radius * math.sin(radians))

    return radians


def start_game(start_button_position_x, start_button_position_y):

    time.sleep(3)
    pyautogui.click(start_button_position_x, start_button_position_y)
    time.sleep(0.1)
    move_to_radians(0, 0)


def get_direction():
    x, y = pyautogui.position()

    return math.atan2(y, x)


def Reward(prev_length, cur_length):
    dif = cur_length - prev_length

    return dif / 100


def screenshot(x, y, w, h, gray, reduction_factor):
    with mss.mss() as sct:
        # The screen part to capture
        region = {'left': x, 'top': y, 'width': w, 'height': h}

        # Grab the data
        img = sct.grab(region)

        if gray:
            result = cv2.cvtColor(np.array(img), cv2.COLOR_BGRA2GRAY)
        else:
            result = cv2.cvtColor(np.array(img), cv2.COLOR_BGRA2BGR)

        img = result[::reduction_factor, ::reduction_factor]
        img = Image.fromarray(img)
        # img.show()
        return img


def read_score(driver):

    dead = False
    score = 10

    try:
        score = int(driver.find_elements_by_tag_name('span')[32].text)
        print('Alive: {}'.format(score))
    except:
        dead = True
        print('Dead')
        pass

    return score, dead

class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, arg):
        """Saves a transition."""
        if len(self.memory) < self.capacity and self.position == len(self.memory):
            self.memory.append(None)
        self.memory[self.position] = arg
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def delete(self):
        
        self.position = (self.position - 1) % self.capacity

        del self.memory[self.position]

        

    def __len__(self):
        return len(self.memory)

def train_model(env_memory, reward_memory, target_memory, action_memory, dqn, target_dqn, lr):

    batch_size = 128

    if len(env_memory) != len(reward_memory):
        sys.exit('Replay Memory error')

    if len(reward_memory) < batch_size * 2:
        return None

    dqn.train()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(dqn.parameters(), lr=lr)
    
    indices = random.sample(range(len(env_memory)), batch_size)
    env_list = [env_memory.memory[i] for i in indices]
    reward_list = [reward_memory.memory[i] for i in indices]
    target_list = [target_memory.memory[i] for i in indices]
    action_list = [action_memory.memory[i] for i in indices]

    env = torch.stack(env_list)
    reward = torch.tensor(reward_list, dtype=torch.float32).view(-1,1)
    reward = reward.to(device)
    
    Q_target = torch.stack(target_list)
    Q_target = Q_target.view(-1, 1)
    
    label = reward + discount_factor * Q_target
    loss_total = 0.

    pred = dqn(env)
    pred = pred[torch.arange(pred.shape[0]), action_list]
    pred = pred.view(-1, 1)

    loss = criterion(pred, label)

    optimizer.zero_grad()
    loss.backward()

    optimizer.step()
    loss_total += loss.detach().item()
    print('loss is {}'.format(loss_total))
    return loss_total

    
if __name__ == "__main__":

    import time
    import pickle
    import numpy
    from numpy import asarray
    import torch.optim as optim
    import warnings
    import torch
    import matplotlib.pyplot as plt

    warnings.filterwarnings("ignore")
    args = parser()

    device = 'cuda'
    dqn = DQN()
    #dqn.load_state_dict(torch.load('Model'))
    dqn = dqn.to(device)
    target_dqn = DQN()
    target_dqn.load_state_dict(dqn.state_dict())
    target_dqn.eval()
    target_dqn.to(device)
    cur_length = 10
    prev_length = 10
    dead = False
    
    max_memory = 10000

    env_memory = ReplayMemory(max_memory)
    reward_memory = ReplayMemory(max_memory)
    target_memory = ReplayMemory(max_memory)
    action_memory = ReplayMemory(max_memory)

    loss_list = []
    final_score_list = []

    n = 0

    width = 1300
    height = 800
    driver = open_and_size_browser_window(width=width, height=height)

    start_game(1306, 228) 
    start_game(722, 600)
    time.sleep(1)

    final_score = 0
    
    score = 10
    prev_score = 10

    epsilon = 0.8
    epoch = 0
    count = 0
    lr = args.lr
    episode = 0
    first = True

    while True:
        
        if episode == args.episode:
            break
        
        
        score, dead = read_score(driver)
        
        if count >= 20:
            driver.close()
            count = 0
            driver = open_and_size_browser_window(width=width, height=height)
            start_game(1306, 228) 
            start_game(722, 600)
            time.sleep(1)


        if dead == True:

            count += 1
            try:
                
                final_score = int(driver.find_element_by_tag_name('b').text)
                final_score_list.append(final_score)

                if len(env_memory)!= 0:
                    
                    # for delay
                    env_memory.delete()
                    reward_memory.delete()
                    action_memory.delete()
                    target_memory.delete()

                    reward_memory.push(-0.5) # reward when died
                    tensor_zero = torch.tensor(0).float().to(device)
                    target_memory.push(tensor_zero)
                    
                print('-----episode is {}-----'.format(episode))
                print("-----Epoch is {}-----".format(epoch))
                print('Final score is {}'.format(final_score))
                driver.close()
                count = 0
                episode += 1

                if episode % 10 == 9:
                    target_dqn.load_state_dict(dqn.state_dict())
                    lr = lr * 0.9
                    epsilon = epsilon * 0.9
                    print('-----lr is {}-----'.format(lr))
                    print('-----epsilon is {}-----'.format(epsilon))

                driver = open_and_size_browser_window(width=width, height=height) 
                start_game(722, 600)
                first=True
                time.sleep(1)
                
            except:
                time.sleep(0.1)

        else:
            epoch += 1
            count = 0

            reward = Reward(prev_score, score)

            if first == False:
                reward_memory.push(reward)

            prev_score = score
            #print('score is {}'.format(score))


            env = screenshot(80,170,1250,650,1,4)
            env = np.asarray(env)
            env = env[np.newaxis,np.newaxis,:,:]
            env = torch.from_numpy(env).float().to(device)
            
            with torch.no_grad():
                Q = dqn(env)

                if first == False:
                    Q_target = torch.max(target_dqn(env)[0])
                    target_memory.push(Q_target)

                if np.random.random() < epsilon:
                    #action_number = np.random.randint(0,16)
                    action_number = np.random.randint(0, 8)
                else:    
                    action_number = torch.argmax(Q)

                action_memory.push(action_number)
                #action(action_number // 2, action_number % 2)
                action(action_number, 0)

            loss = train_model(env_memory, reward_memory, target_memory, action_memory, dqn, target_dqn, lr)
            env_memory.push(env.view(1, 163, 313))  # for same length

            if loss != None:
                loss_list.append(loss)
                
            first = False
            
        time.sleep(0.4)

    driver.close()
    with open ('loss', 'wb') as f:
        pickle.dump(loss_list, f)

    with open('score', 'wb') as f:
        pickle.dump(final_score_list, f)

    torch.save(dqn.state_dict(), 'Model')
