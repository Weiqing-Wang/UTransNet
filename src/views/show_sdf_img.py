import pickle

import numpy as np
from matplotlib import pyplot as plt

input_data=pickle.load(open('../../data/dataX.pkl','rb'))
min1=np.min(input_data[0,0,:,:])
max1=np.max(input_data[0,0,:,:])
min2=np.min(input_data[0,1,:,:])
max2=np.max(input_data[0,1,:,:])
min3=np.min(input_data[0,2,:,:])
max3=np.max(input_data[0,2,:,:])
print(min1,max1)
print(min2,max2)
print(min3,max3)
# data=input_data[0,0,:,:]+input_data[0,1,:,:]+input_data[0,2,:,:]
data=input_data[0,0,:,:]

min=np.min(data)
max=np.max(data)
print(min,max)
#
nx = input_data.shape[2]
ny = input_data.shape[3]

plot_options = {'cmap': 'viridis', 'origin': 'lower', 'extent': [0,nx,0,ny]}

plt.figure()
fig = plt.gcf()
fig.set_size_inches(15, 10)
plt.title('img', fontsize=18)
plt.imshow(np.transpose(data[:, :]), vmin = 0, vmax = max, **plot_options)
plt.colorbar(orientation='horizontal')
plt.show()