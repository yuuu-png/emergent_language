import matplotlib.pyplot as plt
import japanize_matplotlib
import numpy as np

max_token_len = [2, 4, 8, 16, 32]
lstm_MAE = [0.0831, 0.06353, 0.04582, 0.04094, 0.04166]
lstm_avg_token_len = [1.793, 3.744, 7.378, 14.618, 25.951]
lstm_id_performance = [0.7033, 0.8576, 0.9310, 0.9418, 0.9470]
gru_MAE = [0.08142, 0.06513, 0.04786, 0.03648, 0.03956]
gru_avg_token_len = [1.742, 3.651, 7.86, 15.549, 31.243]
gru_id_performance = [0.7375, 0.8493, 0.9264, 0.9549, 0.9596]
transformer_MAE = [0.07994, 0.06331, 0.0455, 0.03213, 0.03027]
transformer_avg_token_len = [1.867, 3.558, 7.223, 14.672, 21.672]
transformer_id_performance = [0.7740, 0.8556, 0.9326, 0.9761, 0.9656]

plt.rcParams["font.size"] = 24

fig = plt.figure(figsize=(10, 7))
plt.plot(max_token_len, lstm_MAE, "o-", label="LSTM")
plt.plot(max_token_len, gru_MAE, "s-", label="GRU")
plt.plot(max_token_len, transformer_MAE, "x-", label="Transformer")
plt.legend(loc="upper right")
plt.xscale("log")
plt.xlabel("最大トークン長")
plt.ylabel("MAE")
plt.savefig("outputs/model_mae.svg")
plt.close()


fig = plt.figure(figsize=(10, 7))
plt.plot(max_token_len, lstm_avg_token_len, "o-", label="LSTM")
plt.plot(max_token_len, gru_avg_token_len, "s-", label="GRU")
plt.plot(max_token_len, transformer_avg_token_len, "x-", label="Transformer")
plt.xticks(max_token_len)
plt.xscale("log")
plt.yscale("log")
plt.legend(loc="upper left")
plt.xlabel("最大トークン長")
plt.ylabel("平均トークン長")
plt.savefig("outputs/model_avg_token_len.svg")
plt.close()


fig = plt.figure(figsize=(10, 7))
plt.plot(max_token_len, lstm_id_performance, "o-", label="LSTM")
plt.plot(max_token_len, gru_id_performance, "s-", label="GRU")
plt.plot(max_token_len, transformer_id_performance, "x-", label="Transformer")
plt.xticks(max_token_len)
plt.xscale("log")
plt.legend(loc="upper left")
plt.xlabel("最大トークン長")
plt.ylabel("識別性能")
plt.savefig("outputs/model_id_performance.svg")
plt.close()
