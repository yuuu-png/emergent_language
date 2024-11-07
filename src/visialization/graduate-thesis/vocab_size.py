import matplotlib.pyplot as plt
import japanize_matplotlib
import numpy as np

max_token_len = [2, 3, 5, 9, 17, 33, 65, 129]
MAE = [0.1111, 0.07793, 0.04925, 0.03971, 0.03139, 0.02787, 0.02468, 0.02478]
avg_token_len = [2.361, 5.759, 9.378, 9.635, 9.924, 9.958, 9.944, 9.94]
id_performance = [0.3646, 0.7787, 0.9282, 0.9500, 0.9685, 0.9702, 0.9745, 0.9787]

plt.rcParams["font.size"] = 24


fig = plt.figure(figsize=(10, 7))
plt.plot(max_token_len, MAE, "o-")
plt.xscale("log")
plt.xlabel("語彙数")
plt.ylabel("MAE")
plt.savefig("outputs/vocab_size_mae.svg")
plt.close()


fig = plt.figure(figsize=(10, 7))
plt.plot(max_token_len, avg_token_len, "o-")
plt.xscale("log")
plt.xlabel("語彙数")
plt.ylabel("平均トークン長")
plt.savefig("outputs/vocab_size_len.svg")
plt.close()


fig = plt.figure(figsize=(10, 7))
plt.plot(max_token_len, id_performance, "o-")
plt.xscale("log")
plt.xlabel("語彙数")
plt.ylabel("識別性能")
plt.savefig("outputs/vocab_sizeance.svg")
plt.close()
