import matplotlib.pyplot as plt
import japanize_matplotlib
import numpy as np

max_token_len = [2, 4, 8, 16, 32, 64, 128]
MAE = [0.07994, 0.06312, 0.0455, 0.03213, 0.03027, 0.01694, 0.02415]
avg_token_len = [1.867, 3.558, 7.223, 14.672, 21.672, 63.009, 101.75]
id_performance = [0.7740, 0.8556, 0.9326, 0.9761, 0.9656, 0.9853, 0.9812]

plt.rcParams["font.size"] = 24

fig = plt.figure(figsize=(10, 7))
plt.plot(max_token_len, MAE, "o-")
plt.xscale("log")
plt.xlabel("最大トークン長")
plt.ylabel("MAE")
plt.savefig("outputs/max_token_len_mae.svg")
plt.close()


fig = plt.figure(figsize=(10, 7))
plt.plot(max_token_len, avg_token_len, "o-")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("最大トークン長")
plt.ylabel("平均トークン長")
plt.savefig("outputs/max_token_len_avg_token_len.svg")
plt.close()


fig = plt.figure(figsize=(10, 7))
plt.plot(max_token_len, id_performance, "o-")
plt.xscale("log")
plt.xlabel("最大トークン長")
plt.ylabel("識別性能")
plt.savefig("outputs/max_token_len_id_performance.svg")
plt.close()
