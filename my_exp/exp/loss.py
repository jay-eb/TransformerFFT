import matplotlib
import matplotlib.pyplot as plt


activation_functions = ['gelu', 'relu', 'leakyRelu']

# 绘制损失曲线
plt.figure(figsize=(8, 6))
for act in activation_functions:
    plt.plot(range(1, len(loss_history[act])+1), loss_history[act], marker='o', label=act)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("不同激活函数下训练过程的损失变化")
plt.legend()
plt.grid(True)
plt.savefig("loss.png")