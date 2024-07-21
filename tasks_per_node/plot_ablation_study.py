import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams["font.family"] = "STIXGeneral"
matplotlib.rcParams["font.size"] = 14


fp32_accuracy = 81.84
fp32_accuracy_std = 0.53
file_path = "ablation_study/Cora/hidden_128/wd_0.0002/lr_0.01/bit_width_2,4,8/results.csv"


if __name__ == "__main__":
    df = pd.read_csv(file_path)
    df_sorted = df.sort_values(by="bit_width_lambda")
    fig, axs = plt.subplots(1, 2, figsize=(6, 2.75))

    axs[0].plot(df_sorted["bit_width_lambda"],
                df_sorted["average_bit_width_mean"],
                color="#0065a7",
                marker=".",
                linestyle="--",
                label="Average Bit Width")
    axs[0].fill_between(df_sorted["bit_width_lambda"],
                        df_sorted["average_bit_width_mean"] - df_sorted["average_bit_width_std"],
                        df_sorted["average_bit_width_mean"] + df_sorted["average_bit_width_std"],
                        color="gray",
                        alpha=0.2,
                        label="Std. Bit Width")
    axs[0].set_xlabel("$\lambda$ \n(a) Effect of $\lambda$ on \naverage bit-width")
    axs[0].set_ylabel("Average Bit Width")
    # axs[0].legend()
    axs[0].grid(True)

    axs[1].scatter(df_sorted["average_bit_width_mean"],
                   df_sorted["accuracy_mean"],
                   marker=".",
                   label="MixQ",
                   color="#0065a7")

    axs[1].axhline(y=fp32_accuracy, color="#885078", linestyle="--", label="FP32")
    axs[1].fill_between([axs[1].get_xlim()[0], axs[1].get_xlim()[1]],
                        fp32_accuracy - fp32_accuracy_std,
                        fp32_accuracy + fp32_accuracy_std,
                        color="#885078",
                        alpha=0.2)

    axs[1].set_xlabel("Average Bit Width \n(b) Accuracy of \nchosen architectures")
    axs[1].set_ylabel("Accuracy (%)")
    axs[1].legend(loc="lower right")
    axs[1].grid(True)

    plt.tight_layout()
    plt.savefig("cora_ablation_study.pdf", bbox_inches="tight")
    plt.show()