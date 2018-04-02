import numpy as np
import pickle

with open('sum_rew_final_policy.pkl','rb') as f:
	li = pickle.load(f)


ch = np.array(li)
catastrophes = np.sum(ch<-150)
opt = np.sum((ch>(max(li)-7))&(ch <= max(li)))
print('min rew', min(li))
print('max rew', max(li))
print('mean rew',np.mean(ch))
print('number of opt episodes', opt)
print('number of catastrophes', catastrophes)
print('percent catastrophes', catastrophes/len(li))
