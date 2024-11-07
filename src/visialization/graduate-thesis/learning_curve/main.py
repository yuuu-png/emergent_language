import csv
import numpy as np
import matplotlib.pyplot as plt

import japanize_matplotlib


def main():
    plt.rcParams["font.size"] = 18

    plot_learning_curve(
        path_csv="/work/src/visialization/learning_curve/max_token_len_len.csv",
        labels=["8", "16", "32", "64", "128"],
        indices=[16, 28, 22, 4, 10],
        y_label="平均トークン長",
        output_path="/work/outputs/learning_curve/max_token_len_len.svg",
    )

    plot_learning_curve(
        path_csv="/work/src/visialization/learning_curve/max_token_len_loss.csv",
        labels=["8", "16", "32", "64", "128"],
        indices=[16, 28, 22, 4, 10],
        y_label="損失",
        output_path="/work/outputs/learning_curve/max_token_len_loss.svg",
    )

    plot_learning_curve(
        path_csv="/work/src/visialization/learning_curve/vocab_size_avg_len.csv",
        labels=["9", "17", "33", "65", "129"],
        indices=[4, 10, 16, 22, 28],
        y_label="平均トークン長",
        output_path="/work/outputs/learning_curve/vocab_size_avg_len.svg",
    )

    plot_learning_curve(
        path_csv="/work/src/visialization/learning_curve/vocab_size_loss.csv",
        labels=["9", "17", "33", "65", "129"],
        indices=[4, 10, 16, 22, 28],
        y_label="損失",
        output_path="/work/outputs/learning_curve/vocab_size_loss.svg",
    )


def plot_learning_curve(
    path_csv: str,
    indices: list,
    labels: list,
    y_label: str,
    output_path: str,
    show_indices: bool = False,
):
    rows = []
    with open(path_csv) as f:
        reader = csv.reader(f)
        for row in reader:
            # 空のセルをNaNに置き換える
            rows.append([x if x != "" else "NaN" for x in row])

    header = rows.pop(0)

    if show_indices:
        for i, e in enumerate(header):
            print(i, e)
        return

    # 文字列の'NaN'を実際のNaN値に変換します
    data = np.array(rows, dtype=np.float_).T

    fig, ax = plt.subplots(figsize=(10, 7))

    zorder = 100
    for index, label in zip(indices, labels):
        ax.plot(
            data[0],
            data[index],
            linestyle="solid",
            marker=",",
            label=label,
            zorder=zorder,
        )
        zorder -= 1

    ax.set_xlabel("Epoch")
    ax.set_ylabel(y_label)
    ax.legend()

    plt.savefig(output_path)


if __name__ == "__main__":
    main()
