import numpy as np
import matplotlib.pyplot as plt

# =========================
# 1. Training curve data
# 来自你训练日志的真实结果
# =========================

train_acc = [
    0.3520, 0.5918, 0.6732, 0.7343, 0.7811,
    0.8155, 0.8355, 0.8690, 0.8765, 0.9003,
    0.9093, 0.9127, 0.9207, 0.9298, 0.9379,
    0.9474, 0.9418, 0.9479, 0.9520, 0.9526,
    0.9519, 0.9576, 0.9551, 0.9520, 0.9637,
    0.9664, 0.9664, 0.9615, 0.9717, 0.9578
]

val_acc = [
    0.5319, 0.6464, 0.7244, 0.7810, 0.8031,
    0.8146, 0.8339, 0.8626, 0.8554, 0.8833,
    0.8848, 0.8840, 0.8776, 0.8862, 0.8905,
    0.8912, 0.8948, 0.8984, 0.8941, 0.8869,
    0.8740, 0.8991, 0.9026, 0.8869, 0.8998,
    0.9041, 0.9048, 0.9069, 0.9098, 0.8962
]

epochs = range(1, 31)

plt.figure(figsize=(8, 5))
plt.plot(epochs, train_acc, marker="o", label="Training accuracy")
plt.plot(epochs, val_acc, marker="o", label="Validation accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Model Accuracy over Epochs")
plt.ylim(0.3, 1.0)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("training_curve.png", dpi=300)
plt.show()


# =========================
# 2. Confusion matrix data
# 来自你打印出来的真实 confusion matrix
# =========================

classes = [
    "air_conditioner",
    "car_horn",
    "children_playing",
    "dog_bark",
    "drilling",
    "engine_idling",
    "gun_shot",
    "jackhammer",
    "siren",
    "street_music"
]

cm = np.array([
    [189, 0,   0,   0,   1,   3,  0,   4,   2,   1],
    [0,   79,  0,   1,   2,   1,  0,   1,   0,   2],
    [2,   1,   159, 17,  3,   4,  0,   0,   3,   11],
    [3,   4,   5,   175, 3,   2,  0,   0,   4,   4],
    [1,   0,   0,   1,   189, 0,  0,   6,   1,   2],
    [1,   1,   1,   1,   2,   187,1,   1,   2,   3],
    [0,   0,   0,   1,   0,   0,  73,  1,   0,   0],
    [4,   2,   0,   0,   10,  0,  1,   183, 0,   0],
    [2,   0,   0,   3,   1,   2,  0,   1,   177, 0],
    [5,   1,   18,  2,   2,   3,  0,   4,   8,   157]
])

plt.figure(figsize=(10, 8))

plt.imshow(cm, cmap="Blues")  # ← 改这里！

plt.title("Confusion Matrix", fontsize=14)
plt.xlabel("Predicted label", fontsize=12)
plt.ylabel("True label", fontsize=12)

plt.xticks(np.arange(len(classes)), classes, rotation=45, ha="right")
plt.yticks(np.arange(len(classes)), classes)

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]),
                 ha="center", va="center",
                 color="black", fontsize=9)


cbar = plt.colorbar()
cbar.ax.set_ylabel("Number of samples", rotation=270, labelpad=15)

plt.tight_layout()
plt.savefig("confusion_matrix_clean.png", dpi=300)
plt.show()