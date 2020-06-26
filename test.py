import numpy as np
import torch
import random

a = [1,2,3, None, 1, None]
if a[-1]==None:
    del a[1]
print(a)