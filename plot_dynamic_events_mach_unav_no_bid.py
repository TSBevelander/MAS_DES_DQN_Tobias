import pandas as pd
from matplotlib import pyplot as plt, rcParams
import numpy as np
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

utilization = [85, 85, 90, 90, 95, 95]
due_date_settings = [4, 6, 4, 6, 4, 6]

fig, ax = plt.subplots(2, 3)
Names = ["0.00", "0.05", "0.10", "0.15", "0.20"]

for i in range(len(utilization)):
    filename = 'Results/Machine_Unav_' + str(utilization[i]) + '_' + str(due_date_settings[i]) + '_No_Bid' +'.csv'
    filename1 = 'Results/Machine_Unav_' + str(utilization[i]) + '_' + str(due_date_settings[i]) + '_Static' + '.csv'

    data = pd.read_csv(filename, index_col=0)
    df = pd.DataFrame(data)

    data = pd.read_csv(filename1, index_col=0)
    df1 = pd.DataFrame(data)

    ax[0, 0].plot(df['Robustness'] / max(df1['Robustness']), 'o-')
    ax[0, 0].set_xlabel("Unavailable Probability/ Repair Time")
    ax[0, 0].set_ylabel("Relative objective vs. RSS")
    ax[0, 0].legend(["85-4", "85-6", "90-4", "90-6", "95-4", "95-6"])
    ax[0, 0].grid(b=True, which='both')
    ax[0, 0].set_title('Robustness')

    ax[0, 1].plot(df['Stability 1'] / max(df1['Stability 1']), 'o-')
    ax[0, 1].set_xlabel("Unavailable Probability/ Repair Time")
    ax[0, 1].set_ylabel("Relative increase in objective vs. RSS")
    ax[0, 1].legend(["85-4", "85-6", "90-4", "90-6", "95-4", "95-6"])
    ax[0, 1].grid(b=True, which='both')
    ax[0, 1].set_title('Stability measure 1')

    ax[0, 2].plot(df['Stability 2'], 'o-')
    ax[0, 2].set_xlabel("Unavailable Probability/ Repair Time")
    ax[0, 2].set_ylabel("Increase in processing time")
    ax[0, 2].legend(["85-4", "85-6", "90-4", "90-6", "95-4", "95-6"])
    ax[0, 2].grid(b=True, which='both')
    ax[0, 2].set_title('Stability measure 2')

    ax[1, 0].plot(df['Stability 3'], 'o-')
    ax[1, 0].set_xlabel("Unavailable Probability/ Repair Time")
    ax[1, 0].set_ylabel("Increase in setup time")
    ax[1, 0].legend(["85-4", "85-6", "90-4", "90-6", "95-4", "95-6"])
    ax[1, 0].grid(b=True, which='both')
    ax[1, 0].set_title('Stability measure 3')

    ax[1, 1].plot(df['Stability 4'], 'o-')
    ax[1, 1].set_xlabel("Unavailable Probability/ Repair Time")
    ax[1, 1].set_ylabel("Increase in job switches")
    ax[1, 1].legend(["85-4", "85-6", "90-4", "90-6", "95-4", "95-6"])
    ax[1, 1].grid(b=True, which='both')
    ax[1, 1].set_title('Stability measure 4')

plt.setp(ax, xticks=[0, 3, 4, 7, 8, 11], xticklabels=['0.05/1', '0.20/1', '0.05/5', '0.20/5', '0.05/10', '0.20/10'])
fig.suptitle('Machine unavailability with no bidding allowed', fontsize=24)
# plt.legend(["85-4", "85-6", "90-4", "90-6", "95-4", "95-6"])
plt.show()

