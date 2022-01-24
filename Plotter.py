import string

import pandas
import pandas as pd
from matplotlib import pyplot as plt, rcParams
import numpy as np
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from scipy.interpolate import make_interp_spline, BSpline
from scipy.ndimage import gaussian_filter1d
from math import factorial
from scipy.signal import savgol_filter

init_data = np.zeros((500, 3, 1))
plot_data_mean = np.zeros((500, 3))
plot_data_std = np.zeros((6, 500))

skip_bid = [7, [0, 1], [0, 3], [0, 4], [0, 5], [0, 7], [0, 7], [1, 3], [1, 4], [1, 5], [1, 7], [1, 7], [3, 4], [3, 5],
            [3, 7], [3, 7], [4, 5], [4, 7], [4, 7],
            [5, 7], [5, 7], [7, 7]]
skip_seq = [3, 3, 3, 3, 3, 0, 1, 3, 3, 3, 0, 1, 3, 3, 0, 1, 3, 0, 1, 0, 1, [8, 9]]

mean = np.zeros((2, 1000))

str1 = []

str1.append("Runs/Attribute_Runs/90-6/Run-90-6-1000-[5, 7]-[3, 3].txt")
str1.append("Runs/Attribute_Runs/90-6/Run-90-6-1000-[7, 7]-[3, 3]-Relearn.txt")

# print(content_list)

for k in range(len(str1)):
    j = 0
    my_file1 = open(str1[k])
    content_list = my_file1.readlines()
    for i in range(len(content_list)):
        output = str.split(content_list[i], " ")
        print(output)
        if output[0]:
            if (float(output[1]) > 300) & (int(output[0]) > 20):
                mean[k][j] = mean[k][j - 1]
            elif (float(output[1]) > 250) & (int(output[0]) < 20):
                mean[k][j] = 250 - int(output[0]) * 2

            else:
                mean[k][j] = float(output[1])
            j += 1
print(mean)


fig = plt.figure()

ax = fig.add_subplot(111)

x = range(1, 1001)
xnew = np.linspace(1, 1000, 100)
for j in range(1):
    # spl = make_interp_spline(x, mean[0][:], k=3)
    # power_smooth = spl(xnew)
    # ax.plot(xnew, power_smooth, linewidth=2)
    #
    # spl = make_interp_spline(x, mean[1][:], k=2)
    # power_smooth = spl(xnew)
    # ax.plot(xnew, power_smooth, linewidth=2)

    yhat = savgol_filter(mean[1][:], 51, 2)
    ax.plot(x, yhat, linewidth=2)

    yhat = savgol_filter(mean[0][:], 51, 2)
    ax.plot(x, yhat, linewidth=2)

ax.xaxis.set_major_locator(MultipleLocator(100))
ax.yaxis.set_major_locator(MultipleLocator(10))
ax.grid(which='major', color='#CCCCCC', linestyle='--')

plt.xlabel('Population number')
plt.ylabel('Mean objective of population')
plt.title('Comparisson with initializing with parameters from other settings')
plt.legend(['Old parameters', 'Initial parameters'])

plt.show()