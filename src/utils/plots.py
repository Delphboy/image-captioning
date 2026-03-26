import matplotlib.pyplot as plt


# TODO: Fix so that the x-axis starts at 1
def plot_training_charts(train_losses, val_losses, val_scores, exp_name: str):
    val_scores_cider = [s["CIDEr"] * 100 for s in val_scores]
    val_scores_bleu1 = [s["BLEU"][0] * 100 for s in val_scores]
    val_scores_bleu4 = [s["BLEU"][3] * 100 for s in val_scores]

    # Plot losses
    plt.figure()
    plt.plot(train_losses, label="Training loss")
    plt.plot(val_losses, label="Validation loss")
    plt.legend()
    plt.title(f"Losses for {exp_name.replace('-', ' ').replace('_', ' ')}")
    plt.savefig(f"{exp_name}-losses.png", dpi=1200)

    # clear figure
    plt.clf()
    plt.plot(val_scores_cider, label="Validation CIDEr")
    plt.plot(val_scores_bleu1, label="Validation BLEU-1")
    plt.plot(val_scores_bleu4, label="Validation BLEU-4")
    plt.legend()
    plt.title(f"Scores for {exp_name.replace('-', ' ').replace('_', ' ')}")
    plt.savefig(f"{exp_name}-evals.png", dpi=1200)
