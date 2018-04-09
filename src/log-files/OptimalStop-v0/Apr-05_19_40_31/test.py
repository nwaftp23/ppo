import numpy as np



def running_mean_std(dat, old_mean, old_var, n):
	m = len(dat)
	new_data_var = np.var(dat)
	new_data_mean = np.mean(dat)
	new_mean = ((old_mean * n) + (new_data_mean * m)) / (m + n)
	varss = ((n * (old_var + old_mean**2)) + (m * (new_data_var + new_data_mean**2))) / (m + n) - new_mean**2
	return new_mean , varss


ee = np.random.normal(0,1,100)
old_mean = np.mean(ee)
old_var = np.var(ee)
print(ee.shape)
n = len(ee)
eee = np.random.normal(0,5,10)
ch = running_mean_std(eee, old_mean, old_var, n)
tot_dat = np.append(ee,eee)
real_mean = np.mean(tot_dat)
real_var = np.var(tot_dat)

print('the real mean is', real_mean)
print('the running mean is', ch[0])
print('Do they match', real_mean == ch[0])
print('the real var is', real_var)
print('the running var is', ch[1])
print('Do they match', real_var == ch[1])