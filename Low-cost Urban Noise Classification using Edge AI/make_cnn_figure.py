import matplotlib.pyplot as plt

plt.figure(figsize=(6, 9))  # 稍微拉高一点

layers = [
    "Input\n(40×174 MFCC)",
    "Conv2D (32)\n+ ReLU\n+ MaxPool",
    "Conv2D (64)\n+ ReLU\n+ MaxPool",
    "Conv2D (128)\n+ ReLU\n+ MaxPool",
    "Flatten",
    "Dense (128)\n+ Dropout",
    "Output\n(10 classes)"
]

spacing = 1.5
y_positions = [i * spacing for i in range(len(layers))][::-1]

box_height = 0.8

# 画框
for i, layer in enumerate(layers):
    plt.text(0.5, y_positions[i], layer,
             ha='center', va='center',
             fontsize=10,
             bbox=dict(
                 boxstyle="round,pad=0.6",
                 facecolor="#E3F2FD",
                 edgecolor="#1E88E5",
                 linewidth=1.5
             ))

# 画箭头
for i in range(len(layers)-1):
    plt.annotate("",
        xy=(0.5, y_positions[i+1] + box_height/2 + 0.1),
        xytext=(0.5, y_positions[i] - box_height/2 - 0.1),
        arrowprops=dict(
            arrowstyle="-|>",
            linewidth=1.5,
            color="#1E88E5"
        )
    )

plt.xlim(0, 1)
plt.ylim(-1, max(y_positions)+1)
plt.axis('off')

plt.title("CNN Architecture for Urban Sound Classification",
          fontsize=13, pad=20)

plt.savefig("cnn_architecture_final.png", dpi=300, bbox_inches='tight')
plt.show()