#!/usr/bin/env python3

import matplotlib.pyplot as plt
from scipy import optimize
import numpy as np
import time

# Load data and the corresponding kernel

indx = 0

for indx in range(154):
  data_file = 'data/data-' + str(indx) + '.npy'
  kernel_file = 'kernel/kernel-' + str(indx) + '.npy'

  data = np.load(data_file)
  kernel = np.load(kernel_file)

  # Solve the linear system
  #sig = optimize.nnls(kernel, data)[0][::-1]
  sig = optimize.nnls(kernel, data)[0]

  # output matrix shapes and residual
  residual = np.linalg.norm(kernel.dot(sig)-data)
  
  print (str(indx), ':', data.shape, kernel.shape, sig.shape, residual)

  A_file = 'kernel/A.' + str(indx) + '.txt'
  b_file = 'data/b.' + str(indx) + '.txt'
  x_file = 'solutions/x.' + str(indx) + '.txt'

  np.savetxt(A_file, kernel, fmt='%.12f')
  np.savetxt(b_file, data, fmt='%.12f')
  np.savetxt(x_file, sig, fmt='%.12f')

  #np.savetxt('A.csv', kernel, delimiter=',', fmt='%.12f')
  #np.savetxt('b.csv', data, delimiter=',', fmt='%.12f')
  #np.savetxt('x.csv', sig, delimiter=',', fmt='%.12f')

# Plot results
#plt.figure(figsize=[12, 2])
#plt.step(np.arange(sig.size), sig, color='orangered')
#plt.grid(True)
#plt.tight_layout()
#plt.show()
