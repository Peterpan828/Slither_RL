import pickle
import matplotlib.pyplot as plt


with open('loss' , 'rb') as f:
    loss = pickle.load(f)

with open('score' , 'rb') as f:
    score = pickle.load(f)



loss_avg = list()

for i in range(len(loss)//10):
    sum(loss[0:10])
    loss_avg.append(sum(loss[i*10:i*10+10])/10)

plt.figure()
plt.plot(loss_avg)
plt.show()


plt.figure()
plt.plot(score)
plt.show()