import numpy as np
import pyautogui
import imutils
import cv2
import sys
import traceback
import logging
import time
import random
import math


import mss
import mss.tools
from PIL import Image
import os
from selenium import webdriver
#from Xlib import display, X
from cv2 import resize

import torch
import math

import mss
import mss.tools
from PIL import Image
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import torch.multiprocessing as mulp
from torch.distributions.categorical import Categorical
#from setproctitle import setproctitle as ptitle

from collections import deque
from argument import parser
from optimizer import SharedAdam

import matplotlib.pyplot as plt
import pickle
from segmentation import UNET

device = 'cpu'
gamma = 0.95

unet = UNET(1, 3)
unet.load_state_dict(torch.load('./FCN/saved_models/first.pt', map_location = 'cpu'))

def open_and_size_browser_window(width, height, x_pos=0, y_pos=0, url='http://www.slither.io'):

    # opens the browser window
    chrome_options = webdriver.ChromeOptions()
    chrome_options.add_argument("--disable-infobars")
    chrome_options.add_argument("--disable-device-discovery-notifications")
    chrome_options.add_argument("--disable-default-apps")
    chrome_options.add_argument("--disable-notifications")
    driver = webdriver.Chrome("./chromedriver", chrome_options=chrome_options)
    driver.set_window_size(width, height)

    driver.set_window_position(x_pos, y_pos)
    driver.get(url)

    return driver



class Actor_Critic(nn.Module):

    def __init__(self):
        super(Actor_Critic, self).__init__()
        self.conv1 = nn.Conv2d(2, 32, 5, stride=1, padding=2)
        self.maxp1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 32, 5, stride=1, padding=1)
        self.maxp2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, 4, stride=1, padding=1)
        self.maxp3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.maxp4 = nn.MaxPool2d(2, 2)

        self.lstm = nn.LSTMCell(1024, 512)
        self.fc_critic = nn.Linear(512, 1)
        self.fc_actor = nn.Linear(512, 16)

    def forward(self, inputs):
        x, (hx, cx) = inputs
        x = F.relu(self.maxp1(self.conv1(x)))
        x = F.relu(self.maxp2(self.conv2(x)))
        x = F.relu(self.maxp3(self.conv3(x)))
        x = F.relu(self.maxp4(self.conv4(x)))
        x = x.view(x.size(0), -1)

        hx, cx = self.lstm(x, (hx, cx))

        prob = F.softmax(self.fc_actor(hx), dim=1)
        value = self.fc_critic(hx)

        return prob, value, (hx, cx)

def weights_init_bias(m):

    classname = m.__class__.__name__
   
    if classname.find('Linear') != -1:
        m.weight.data.uniform_(0.0, 1.0)
        m.bias.data.fill_(0)


def preprocess(state):
    img = state.mean(2)
    img = img.astype(np.float32)
    img = (img - img.mean()) / (img.std())
    img = resize(img, (80, 80))
    img = np.reshape(img, [1, 80, 80])
    img = torch.from_numpy(img).unsqueeze(0)

    with torch.no_grad():
        result = unet(img)

    result = result.squeeze(0)
    conv_result = torch.zeros((80,80))

    for i in range (0,80):
        for j in range (0,80):
            if result[0][i][j] > result [1][i][j] and result[0][i][j] > result [2][i][j]:
                conv_result[i][j] = 0
            elif result[1][i][j] > result [0][i][j] and result[1][i][j] > result [2][i][j]:
                conv_result[i][j] = -5
            else:
                conv_result[i][j] = 5

    conv_result = conv_result.unsqueeze(0).unsqueeze(0)
    conv_result = torch.cat((conv_result, img), 1)
    #print(conv_result)
    #time.sleep(5)
    return conv_result


def action(number, click):
    radian = 2 * math.pi * number / 8
    move_to_radians(radian, click=click)


def move_to_radians(radians, click, radius = 100):
    
    if click == 0:
        #pyautogui.moveTo(728 + radius * math.cos(radians)
        #        , 492 + radius * math.sin(radians))
        pyautogui.moveTo(935 + radius * math.cos(radians)
                , 581 + radius * math.sin(radians))
        time.sleep(0.2)

    else:
        #pyautogui.mouseDown(728 + radius * math.cos(radians)
        #        , 492 + radius * math.sin(radians))
        pyautogui.mouseDown(935 + radius * math.cos(radians)
                , 581 + radius * math.sin(radians))
        time.sleep(0.2)
        #pyautogui.mouseUp(728 + radius * math.cos(radians)
        #        , 492 + radius * math.sin(radians))
        pyautogui.mouseUp(935 + radius * math.cos(radians)
                , 581 + radius * math.sin(radians))
    
    return radians


def start_game(start_button_position_x, start_button_position_y):

    time.sleep(1)
    pyautogui.click(start_button_position_x, start_button_position_y)
    time.sleep(0.1)
    move_to_radians(0, 0)


def get_direction():
    x, y = pyautogui.position()
    x -= 935
    y -= 581
    clicked = 0

    radius = pow(x, 2) + pow(y, 2)
    if (radius > 180000):
        clicked = 1
        
    radian = math.atan2(y,x)
    
    if radian < 0:
        radian += 2 * math.pi
    #print(radian)

    action = int(radian * 8 / (2 * math.pi))
    
    return action, clicked


def Reward(prev_length, cur_length):
    dif = cur_length - prev_length  
    return dif


def screenshot(x, y, w, h):
    with mss.mss() as sct:
        # The screen part to capture
        region = {'left': x, 'top': y, 'width': w, 'height': h}

        # Grab the data
        img = sct.grab(region)
        img = cv2.cvtColor(np.array(img), cv2.COLOR_BGRA2BGR)

        return img

def plot_screen(img):

    # for test
    
    plt.figure()
    plt.imshow(img.squeeze(0).squeeze(0))
    plt.show()


def read_score(driver):

    dead = False
    score = 10

    try:
        score = int(driver.find_elements_by_tag_name('span')[32].text)
        #print('Alive: {}'.format(score))
    except:
        dead = True
        #print('Dead')
        pass

    return score, dead


def train(args, global_model, optimizer, score_list):

    gpu_id = 0
    torch.manual_seed(100)

    local_model = Actor_Critic()

    hx = torch.zeros(1, 512)
    cx = torch.zeros(1, 512)

    width = 1250
    height = 650
    driver = open_and_size_browser_window(width=width, height=height)

    #start_game(1306, 228)
    #start_game(722, 600)
    start_game(973, 784)
    time.sleep(1)
    
    score = 10
    prev_score = 10
    
    #epsilon = 0.5
    count = 0
    debug = 0
    
    start_time = time.time()
    record_time = start_time
    local_model.eval()

    final_score_list = score_list
    NLLLoss = nn.NLLLoss()
    losses = []
    clicked = 0

    with open('supervised_loss', 'rb') as f:
        losses = pickle.load(f)

    while True:

        if time.time() > start_time + args.time * 3600:
            with open('supervised_loss', 'wb') as f:
                pickle.dump(losses, f)

            break

        if time.time() > record_time + 0.1 * 3600:
            print("-----Save-----!!")
            record_time = time.time()
            torch.save(global_model.state_dict(), 'model_slither_supervised')
            with open('supervised_loss', 'wb') as f:
                pickle.dump(losses, f)

            #with open('final_score_supervised', 'wb') as f:
            #    pickle.dump(final_score_list, f)
            continue

        local_model.load_state_dict(global_model.state_dict())
        hx = torch.zeros(1, 512)
        cx = torch.zeros(1, 512)

        entropies = []
        values = []
        log_probs = []
        rewards = []
        labels = []
        outputs = []
        is_dead = -1
        
        for step in range(args.step_episode):

            if is_dead != -1:
                break
            
            state = screenshot(20,200,1700,760)
            state = preprocess(state)
            
            #plot_screen(state)

            prob, value, (hx, cx) = local_model((state, (hx, cx)))

            log_prob = torch.log(prob)
            outputs.append(log_prob)
            entropy = -(log_prob * prob).sum(1)

            m = Categorical(prob)
            action_n = m.sample().detach()
            #print(action_n)
            log_prob = log_prob.gather(1, action_n.unsqueeze(0))

            #action(action_n.cpu() // 2, action_n.cpu() % 2)
            
            if (clicked == 1):
                pyautogui.mouseUp()
                clicked = 0

            dir, clicked  = get_direction()
            if (clicked == 1):
                pyautogui.mouseDown()
                time.sleep(0.2)
                
            label = dir * 2 + clicked
            print("label : ", label)
            #time.sleep(1)
            labels.append(label)

            score, dead = read_score(driver)

            if dead == True:
                reward = 0
                count += 1

            else:
                count = 0

            reward = Reward(prev_score, score)
            reward = max(min(reward, 3), -3)
            prev_score = score

            entropies.append(entropy)
            values.append(value)
            log_probs.append(log_prob)
            rewards.append(reward)

            if count >= 20:
                is_dead = 1 
                break

            is_dead = driver.execute_script("return dead_mtm")
            
            if reward == 0:
                debug += 1
            else:
                debug = 0

            # For Penalty
            if debug == 25:
                print('Penalty!')
                time.sleep(2)
                is_dead = 1
                break
        
        if count < 20:
            
            print('Trainig Start')
            
            R = torch.zeros(1, 1)
            gae = torch.zeros(1, 1)

            if is_dead == -1:
                state = screenshot(20,200,1700,760)
                state = preprocess(state)

                _, value, _ = local_model((state, (hx, cx)))
                R = value.detach()

            values.append(R)
            policy_loss = 0
            critic_loss = 0
            
            #print(torch.cat(outputs, dim=0).shape)
            #print(torch.tensor(labels).shape)
            supervised_loss = NLLLoss(torch.cat(outputs, dim=0), torch.tensor(labels))
            #print("Supervised Loss : ", supervised_loss)
            losses.append(supervised_loss.detach()) 

            for i in reversed(range(len(rewards))):
                R = gamma * R + rewards[i]
                A = R - values[i]
                critic_loss = critic_loss + 0.5 * A.pow(2)

                delta = rewards[i] + gamma * values[i + 1].detach() - values[i].detach()
                gae = gae * gamma + delta

                policy_loss = policy_loss - \
                    log_probs[i] * gae - 0.01 * entropies[i]

            local_model.zero_grad()

            #total_loss = policy_loss + 0.5 * critic_loss
            total_loss = 0.5 * critic_loss + supervised_loss
            total_loss.backward()

            for param, global_param in zip(local_model.parameters(), global_model.parameters()):
                global_param._grad = param.grad.cpu()
                
            optimizer.step()
            print('Trainig Finished')

        entropies.clear()
        values.clear()
        log_probs.clear()
        rewards.clear()
        outputs.clear()

        if is_dead != -1:

            time.sleep(3)
            try:
                final_score = int(driver.find_element_by_tag_name('b').text)
            except:
                final_score = 10

            final_score_list.append(final_score)
            
            driver.close()
            
            count = 0
            debug = 0
            driver = open_and_size_browser_window(width=width, height=height)
            #start_game(1306, 228)
            #start_game(722, 600)
            start_game(973, 784) 
            prev_score = 10
            time.sleep(1)
    
    return final_score_list
      
                
def test(args, global_model, test_score):

    #gpu_id = 0
    local_model = Actor_Critic()

    for i in range(args.test):

        final_score = 0

        
        hx = torch.zeros(1, 512)
        cx = torch.zeros(1, 512)

        width = 1250
        height = 650
        driver = open_and_size_browser_window(width=width, height=height)

        #start_game(1306, 228)
        start_game(973, 784) 
        time.sleep(1)

        local_model.eval()

        local_model.load_state_dict(global_model.state_dict())
        hx = torch.zeros(1, 512)
        cx = torch.zeros(1, 512)

        is_dead = -1
        
        while is_dead == -1:

            state = screenshot(20,200,1700,760)
            state = preprocess(state)
            
            #plot_screen(state)

            with torch.no_grad():

                prob, _, (hx, cx) = local_model((state, (hx, cx)))

            action_n = prob.max(1)[1].data.cpu().numpy()


            if args.random == 0:
                action(action_n[0] //2 , action_n[0] % 2)
            
            else:
                action_n = random.randint(0, 15)
                action(action_n // 2, action_n % 2)


            is_dead = driver.execute_script("return dead_mtm")
        

        time.sleep(3)
        final_score = int(driver.find_element_by_tag_name('b').text)

        test_score['Supervised'].append(final_score - 10)

        driver.close()
    
    return test_score

                



if __name__ == "__main__":

    import os
    import time
    import warnings

    warnings.filterwarnings("ignore")

    args = parser()
    mulp.set_start_method('spawn')

    global_model = Actor_Critic()
    global_model.apply(weights_init_bias)
    
    score = []
    test_score = dict()
    test_score['policy'] = []
    test_score['random'] = []

    with open('test_score', 'rb') as f:
        test_score = pickle.load(f)

    test_score['Supervised'] = []
    if args.test != 0:
        global_model.load_state_dict(torch.load('model_slither_supervised'))
        global_model.eval()
        test_score = test(args,global_model, test_score)
        #with open('test_score', 'wb') as f:
        #    pickle.dump(test_score, f)
            
    else:
        global_model.load_state_dict(torch.load('model_slither_supervised'))
        global_model.train()
        optimizer = SharedAdam(global_model.parameters(), lr=1e-3)
        final_score_list = train(args, global_model, optimizer, score)
        torch.save(global_model.state_dict(), 'model_slither_supervised')

        #with open('final_score', 'wb') as f:
        #    pickle.dump(final_score_list, f)
        