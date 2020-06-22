import pickle
import matplotlib.pyplot as plt


with open('loss' , 'rb') as f:
    loss = pickle.load(f)

with open('score' , 'rb') as f:
    score = pickle.load(f)


a = len(score) // 10
score_avg = []

for i in range(a):
    b = 0
    b += sum(score[i*10:i*10+10]) / 10
    score_avg.append(b)


plt.figure()
plt.plot(loss)
plt.show()


plt.figure()
plt.plot(score_avg)
plt.show()