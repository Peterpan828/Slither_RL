import numpy as np
import torch
import random

a = torch.tensor([[1,2],[3,4],[5,6], [7,8]])
indices = random.sample(range(len(a)),2)
print(a[indices])


action_list = [1,2,3,4,5]
indices = [1,2]
print([action_list[i] for i in indices])