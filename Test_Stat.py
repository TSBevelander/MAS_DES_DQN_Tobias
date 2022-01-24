import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats
import itertools

str1 = "Results/Attributes_Final_90-6.csv"
df = pd.read_csv(str1, header=None)
df = df.dropna()
results = df.values.tolist()

# print(results)

print(np.nanmean(results, axis=0))

# skip_bid = [[7, 7], [2, 7], [4, 7], [5, 7]]
# skip_seq = [[3, 3], [3, 3], [3, 3], [3, 3]]
#
skip_bid = [[7, 7], [0, 7], [1, 7], [2, 7], [3, 7], [4, 7], [5, 7], [7, 7], [7, 7], [7, 7]]
skip_seq = [[3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [3, 3], [0, 3], [1, 3], [2, 3]]


def column(matrix, ii):
    return [row[ii] for row in matrix]

new_res = []
for i in range(100):
    new_res.append(column(results, 0)[i] - column(results, 7)[i])

plt.hist(new_res)
plt.show()

pair = []
pair2 = []
for i in range(0, len(skip_seq) - 1):
    print(stats.shapiro(column(results, i)))
    tt, tp = stats.ttest_rel(column(results, 0), column(results, i + 1), axis=0, nan_policy='propagate',
                             alternative='two-sided')
    format_float = "{:.2f}".format((np.mean(column(results, i + 1)) - np.mean(column(results, 0))))
    print(format_float, tp)
    if ((tp < 0.01) & (np.mean(column(results, i + 1)) - np.mean(column(results, 0)) < 0)) | (tp > 0.01):
        if (skip_bid[i + 1][0] < 7) & (skip_bid[i + 1][1] == 7):
            pair.append(skip_bid[i + 1][0])
        else:
            pair.append(skip_seq[i + 1][0] + 8)

# for i in range(0, len(skip_seq) - 1):
#     tt, tp = stats.ttest_rel(column(results, 0), column(results, i + 1), axis=0, nan_policy='propagate',
#                              alternative='two-sided')
#     format_float = "{:.2f}".format((np.mean(column(results, i + 1)) - np.mean(column(results, 0))))
#     print(format_float, tp)
#     if ((tp < 0.01) & (np.mean(column(results, i + 1)) - np.mean(column(results, 0)) < 0)) | (tp > 0.01):
#         # pair.append(skip_bid[i + 1])
#         if (skip_bid[i + 1][0] < 7) & (skip_seq[i + 1][1] < 7):
#             pair.append(skip_bid[i + 1])
#         else:
#             pair_dual = [skip_bid[i + 1][0], skip_seq[i + 1][1] + 8]
#             pair.append(pair_dual)
# #
print(pair)
#
# for i in range(len(pair)):
#     for j in range(i + 1, len(pair)):
#         pair_dual = list(set(pair[i] + (pair[j])))
#         if len(pair_dual) < 4:
#             pair2.append(pair_dual)
#
# print(pair2)
# #
# pair_bid = []
# pair_seq = []
# for i in range(len(pair)):
#     for j in range(i + 1, len(pair)):
#         if (pair[i] < 7) & (pair[j]) < 7:
#             p = [pair[i], pair[j]]
#             p1 = [3, 3]
#         elif (pair[i] > 7) & (pair[j]) > 7:
#             p = [7, 7]
#             p1 = [pair[i] - 8, pair[j] - 8]
#         else:
#             p = [pair[i], 7]
#             p1 = [pair[j] - 8, 3]
#         pair_bid.append(p)
#         pair_seq.append(p1)
# print(pair_bid)
# print(pair_seq)
# #
# # pair2[] = []
# #
# #
# pair3 = []
# for i in range(len(pair2)):
#     for j in range(len(pair)):
#         if all(x in pair2[i] for x in pair[j]):
#             pair3.append(pair2[i])
#
# pair3.sort()
# print(list(pair3 for pair3,_ in itertools.groupby(pair3)))
# # print(pair3)
