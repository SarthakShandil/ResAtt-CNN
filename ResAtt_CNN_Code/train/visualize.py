import matplotlib.pyplot as plt

def plot_loss_curve(loss_list):
    plt.plot(loss_list)
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.grid(True)
    plt.show()

def plot_heatmap(true_csi, pred_csi):
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].imshow(np.abs(true_csi), aspect='auto', cmap='viridis')
    axs[0].set_title("True Channel")
    axs[1].imshow(np.abs(pred_csi), aspect='auto', cmap='viridis')
    axs[1].set_title("Predicted Channel")
    plt.suptitle("Channel Heatmap")
    plt.show()
