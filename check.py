import pickle
import matplotlib.pyplot as plt
from statistics import median

with open('final_score', 'rb') as f:
    score = pickle.load(f)

with open('supervised_loss', 'rb') as f:
    loss = pickle.load(f)

plt.figure()
plt.plot(loss)
plt.show()

"""
with open('test_score', 'rb') as f:
    test_score = pickle.load(f)

test_score['policy'] = test_score['policy'][:10]

print(median(test_score['policy']))
print(median(test_score['random']))

plt.figure()
plt.title('Final Score')
plt.plot(test_score['policy'], label = 'Trained Model')
plt.plot(test_score['random'], label = 'Random Policy')
plt.legend()
plt.savefig('save/test', dpi=300)
plt.show()
"""

#plt.figure()
#plt.plot(score)
#plt.show()

score_avg_100 = list()

for i in range(len(score)-100):
    score_avg_100.append(sum(score[i:i+100])/100)

score_avg_1000 = list()
xlabel = range(900,len(score)-100)

for i in range(len(score)-1000):
    score_avg_1000.append(sum(score[i:i+1000])/1000)

plt.figure()
plt.title("Final Score")    
plt.plot(score_avg_100, label = 'Rolling Window = 100')
plt.plot(xlabel, score_avg_1000, label = 'Rolling Window = 1000')
plt.legend()
plt.savefig('save/score', dpi=300)
plt.show()
