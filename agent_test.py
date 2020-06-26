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
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T



def open_and_size_browser_window(width, height, x_pos=0, y_pos=0, url='http://www.slither.io'):

    #opens the browser window
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
    move_to_radians(radian, click= click)

def move_to_radians(radians, click, radius = 100):
    
    if click == 0:
        pyautogui.moveTo(728 + radius * math.cos(radians)
                , 492 + radius * math.sin(radians))
    else:
        pyautogui.mouseDown(728 + radius * math.cos(radians)
                , 492 + radius * math.sin(radians))
        time.sleep(0.1)
        pyautogui.mouseUp(728 + radius * math.cos(radians)
                , 492 + radius * math.sin(radians))
    
    return radians


def start_game(start_button_position_x, start_button_position_y):

    time.sleep(1)
    pyautogui.click(start_button_position_x, start_button_position_y)
    time.sleep(1)
    pyautogui.click()
    time.sleep(1)
    move_to_radians(0, 0)
    

def get_direction():
    x, y = pyautogui.position()
    
    return math.atan2(y, x)

def Reward(prev_length, cur_length):
    dif = cur_length - prev_length
    reward = 0

    if dif > 10:
        reward = 20

    elif dif > 3:
        reward = 3

    elif dif == 0:
        reward = -1
    
    elif dif < -5 :
        reward = -3

    return reward
    
            
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
            #img.show()
            return img



                    
if __name__ == "__main__":

    import time
    import pickle
    import numpy
    from numpy import asarray
    import torch.optim as optim
    import warnings
    import torch
    warnings.filterwarnings("ignore")
    
    device = 'cuda'
    dqn = DQN()
    dqn.load_state_dict(torch.load('Model'))
    dqn = dqn.to(device)
    dqn.eval()
    cur_length = 10
    prev_length = 10
    dead = False
    discount_factor = 0.98
    action_list = []
    Qvalue_list = []
    env_list = []
    label_list = []
    loss_list = []

    n = 0

    width = 1300
    height = 800
    driver = open_and_size_browser_window(width = width, height = height)

    start_game(722, 600)   
    time.sleep(3)
    
    
    while True:
        
        # Get env
        env = screenshot(80,170,1250,650,1,4)
        env = asarray(env)
        env = env[numpy.newaxis,numpy.newaxis,:,:]
        env = torch.from_numpy(env)
        env = env.type(torch.float32)
        env = env.to(device)

        # Predict Q_value
        Qvalue = dqn(env)
        action_number = torch.argmax(Qvalue)
        action(action_number, 0)


        time.sleep(0.1)
        
        

        

        
            



                    
                    
                   
