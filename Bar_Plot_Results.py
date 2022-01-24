import pandas as pd
from matplotlib import pyplot as plt, rcParams
import numpy as np
from matplotlib.ticker import MultipleLocator, AutoMinorLocator

utilization = [85, 85, 90, 90, 95, 95]
due_date_settings = [4, 6, 4, 6, 4, 6]
fig, ax = plt.subplots(3, 2)
index = np.zeros((8, 6))
filename1 = 'Results/Custom_2.csv'
filename2 = 'Results/Custom_1.csv'
for i in range(6):
    filename = 'Results/Dispatching-' + str(utilization[i]) + '-' + str(due_date_settings[i]) + '-1.csv'

    data = pd.read_csv(filename, index_col=0)
    df = pd.DataFrame(data)

    data = pd.read_csv(filename1, index_col=0)
    df1 = pd.DataFrame(data)

    data = pd.read_csv(filename2, index_col=0)
    df2 = pd.DataFrame(data)

    D = {'Makespan': [], 'Mean Tardiness': [], 'Max Tardiness': [], 'FlowTime': [], 'WIP': [], 'Late Jobs 1': [],
         'Late Jobs 2': [], 'Late Jobs 3': []}

    makespan = [df["Makespan"].min(), df1.get('Makespan')[i], df2.get('Makespan')[i]]
    D['Makespan'] = [makespan[i] / max(makespan) for i in range(len(makespan))]
    index[2, i] = np.argmin(df["Makespan"])

    mean_tardiness = [df["Mean Weighted Tardiness"].min(), df1.get('Mean Weighted Tardiness')[i],
                      df2.get('Mean Weighted Tardiness')[i]]
    D['Mean Tardiness'] = [mean_tardiness[i] / max(mean_tardiness) for i in range(len(mean_tardiness))]
    index[0, i] = np.argmin(df["Mean Weighted Tardiness"])

    max_tardiness = [df["Max Weighted Tardiness"].min(), df1.get('Mean Weighted Tardiness')[i],
                     df2.get('Mean Weighted Tardiness')[i]]
    D['Max Tardiness'] = [max_tardiness[i] / max(max_tardiness) for i in range(len(max_tardiness))]
    index[1, i] = np.argmin(df["Max Weighted Tardiness"])

    flowtime = [df["Mean Flow Time"].min(), df1.get('Mean Flow Time')[i], df2.get('Mean Flow Time')[i]]
    D['FlowTime'] = [flowtime[i] / max(flowtime) for i in range(len(flowtime))]
    index[3, i] = np.argmin(df["Mean Flow Time"])

    wip = [df["Mean WIP"].min(), df1.get('Mean WIP')[i], df2.get('Mean WIP')[i]]
    D['WIP'] = [wip[i] / max(wip) for i in range(len(wip))]
    index[4, i] = np.argmin(df["Mean WIP"])

    # Normalize the Data
    Names = ["SPT", "W-ACTS", "Proposed"]
    width = 0.2
    x = np.arange(5)
    if (i >= 2) & (i < 4):
        j = 1
    elif i >= 4:
        j = 0
    else:
        j = 2
    # print(j)
    k = (2 - i) % 2

    bar1 = ax[j, k].bar(x - 0.2, [D['Mean Tardiness'][0], D['Max Tardiness'][0], D['Makespan'][0], D['FlowTime'][0],
                                  D['WIP'][0]],
                        width,
                        color=["Blue"], hatch='//')

    bar2 = ax[j, k].bar(x, [D['Mean Tardiness'][1], D['Max Tardiness'][1], D['FlowTime'][1], D['Makespan'][1],
                            D['WIP'][1]], width,
                        color=["Red"], hatch='||')
    bar3 = ax[j, k].bar(x + 0.2, [ D['Mean Tardiness'][2], D['Max Tardiness'][2], D['FlowTime'][2], D['Makespan'][2],
                                  D['WIP'][2]],
                        width,
                        color=["Yellow"], hatch='xx')

    ax[j, k].set_xlabel("Criteria")
    ax[j, k].set_ylabel("Normalized Value")
    for f, rect in enumerate(bar1):
        height = rect.get_height()
        ax[j, k].text(rect.get_x() + rect.get_width() / 2.0, height, f'C{index[f, i] + 1:.0f}', ha='center',
                      va='bottom')
    # ax.legend(["Combination", "W-ATCS", "Proposed"])
plt.setp(ax, xticks=range(5), xticklabels=['Mean Tardiness', 'Max Tardiness', 'Makespan', 'Flowtime', 'Mean WIP'])
ax[0, 1].legend(["Combination", "W-ATCS", "Proposed"], loc='upper right', bbox_to_anchor=(1.0, 1.35))

utilization = [0.95, 0.90, 0.85]
due_date_tightness = [6, 4]
for i, row in enumerate(ax):
    for j, cell in enumerate(row):
        print(j)
        if i == len(ax) - 1:
            cell.set_xlabel("Due Date Tightness: {0:d}".format(due_date_tightness[j - 1]), fontsize=14)
        if j == 0:
            # print(i)
            cell.set_ylabel("Utilization: {0:.2f}".format(utilization[i]), fontsize=14)

plt.show()
