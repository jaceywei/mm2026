import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim 

def load_insect_data(file_path):
    """
    读取昆虫分类数据集
    文件格式要求：每一行是 "param1 param2 label"，空格分隔
    """
    try:
        # np.loadtxt 会自动处理空格和换行
        data = np.loadtxt(file_path)
        
        # 切片：前两列是特征 (params)，最后一列是标签 (label)
        # 将 params 保持为浮点型，将 label 转换为整型以便后续输入神经网络
        params = data[:, :2] 
        labels = data[:, 2].astype(int) 
        
        print(f"✅ 成功读取文件: {file_path}")
        print(f"📊 数据总行数: {len(data)}")
        print(f"特征 (Params) 矩阵的形状: {params.shape}")
        print(f"标签 (Labels) 向量的形状: {labels.shape}\n")
        
        return params, labels

    except FileNotFoundError:
        print(f"❌ 错误：找不到文件 '{file_path}'。")
        print("请确保 txt 文件与你的 Python 脚本放在同一个文件夹下，或者提供绝对路径。")
        return None, None

def plot_insect_comparison(dataset_dict):
    """
    横向对比绘制多个数据集的分布图
    :param dataset_dict: 字典格式 {'标题1': (params1, labels1), '标题2': (params2, labels2)}
    """
    n = len(dataset_dict)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    
    # 如果只有一个数据集，把 axes 转成列表方便统一处理
    if n == 1:
        axes = [axes]
        
    colors = ['red', 'blue', 'green']
    markers = ['o', 's', '^']
    classes = ['Class 0', 'Class 1', 'Class 2']

    for ax, (title, (params, labels)) in zip(axes, dataset_dict.items()):
        if params is None or labels is None:
            continue
            
        for i in range(3):
            idx = (labels == i)
            ax.scatter(params[idx, 0], params[idx, 1], 
                       c=colors[i], marker=markers[i], 
                       label=classes[i], alpha=0.6, edgecolors='k')
        
        ax.set_title(title)
        ax.set_xlabel('Param 1 (Body Length)')
        ax.set_ylabel('Param 2 (Wing Length)')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

class InsectClassifier(nn.Module):
    def __init__(self):
        super(InsectClassifier, self).__init__()
        # 定义网络层
        self.network = nn.Sequential(
            # 输入层: 2 个参数 -> 隐藏层 1: 32 个神经元
            nn.Linear(2, 32),
            nn.ReLU(),
            
            # 隐藏层 1 -> 隐藏层 2: 16 个神经元
            nn.Linear(32, 16),
            nn.ReLU(),
            
            # 隐藏层 2 -> 输出层: 3 个类别
            nn.Linear(16, 3)
        )
        
    def forward(self, x):
        # 注意：在使用 nn.CrossEntropyLoss 时，最后一层不需要加 Softmax
        return self.network(x)

def train_and_evaluate(train_file, test_file, epochs=1000, lr=0.01):
    # --- Step 1: 准备数据 ---
    # 使用你写的 load_insect_data 函数
    X_train_np, y_train_np = load_insect_data(train_file)
    X_test_np, y_test_np = load_insect_data(test_file)
    
    if X_train_np is None or X_test_np is None:
        return

    # 转换为 PyTorch Tensors
    X_train = torch.FloatTensor(X_train_np)
    y_train = torch.LongTensor(y_train_np)
    X_test = torch.FloatTensor(X_test_np)
    y_test = torch.LongTensor(y_test_np)

    # --- Step 2: 初始化模型、损失函数和优化器 ---
    model = InsectClassifier()
    criterion = nn.CrossEntropyLoss() # 适用于多分类任务
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 记录损失以便后续分析
    loss_history = []

    # --- Step 3: 训练循环 ---
    print(f"🚀 开始训练 [{train_file}]...")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()           # 清空梯度
        
        outputs = model(X_train)        # 前向传播
        loss = criterion(outputs, y_train)
        
        loss.backward()                 # 反馈传播
        optimizer.step()                # 更新权重
        
        loss_history.append(loss.item())
        
        if (epoch + 1) % 200 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    # --- Step 4: 测试模型 ---
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test)
        _, predicted = torch.max(test_outputs, 1)
        
        # 按照作业要求：前 60 个和后 150 个分别测试
        # 前 60 个：从训练数据中随机抽取（考察拟合能力）
        # 后 150 个：新数据（考察泛化能力）
        acc_first_60 = (predicted[:60] == y_test[:60]).float().mean().item()
        acc_last_150 = (predicted[60:] == y_test[60:]).float().mean().item()
        
        print(f"\n✅ 训练完成！")
        print(f"📊 测试集结果 - 前 60 个数据准确率: {acc_first_60:.2%}")
        print(f"📊 测试集结果 - 后 150 个数据准确率: {acc_last_150:.2%}")
        print("-" * 40)

    # 返回损失历史，方便你画图展示实验结果（加分项！）
    return loss_history, model

def plot_training_loss(loss_dicts, title='Model Training Progress'):
    """
    封装损失曲线绘制
    :param loss_dicts: 可以是一个列表 [loss1, loss2...]，也可以是字典 {'Dataset 1': loss1, 'Dataset 2': loss2}
    """
    plt.figure(figsize=(8, 5))
    
    # 如果传入的是字典，自动带上 Label
    if isinstance(loss_dicts, dict):
        for label, history in loss_dicts.items():
            plt.plot(history, label=label)
    # 如果传入的是单一列表
    elif isinstance(loss_dicts, list) and not isinstance(loss_dicts[0], list):
        plt.plot(loss_dicts, label='Training Loss')
    # 如果传入的是列表的列表
    else:
        for i, history in enumerate(loss_dicts):
            plt.plot(history, label=f'Run {i+1}')

    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss (Cross Entropy)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

def plot_decision_boundary(model, X, y, title="Decision Boundary"):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    grid_tensor = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])
    model.eval()
    with torch.no_grad():
        outputs = model(grid_tensor)
        _, predicted = torch.max(outputs, 1)
        z = predicted.reshape(xx.shape).numpy()

    # --- 关键修改：定义与散点图一致的自定义色板 ---
    from matplotlib.colors import ListedColormap
    custom_colors = ['red', 'blue', 'green']
    custom_cmap = ListedColormap(custom_colors)

    plt.figure(figsize=(8, 6))
    
    # 使用自定义的 cmap，这样 0=红, 1=蓝, 2=绿
    plt.contourf(xx, yy, z, alpha=0.2, cmap=custom_cmap)
    
    # 绘制原始数据点
    for i in range(3):
        idx = (y == i)
        plt.scatter(X[idx, 0], X[idx, 1], c=custom_colors[i], 
                    edgecolors='k', s=25, label=f'Class {i}')

    plt.title(title)
    plt.xlabel('Param 1')
    plt.ylabel('Param 2')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    
    x1, y1 = load_insect_data('insects-training.txt')
    x2, y2 = load_insect_data('insects-2-training.txt')

    # 2. 一行代码展示数据对比
    plot_insect_comparison({
        "Training dataset 1 (Clean)": (x1, y1),
        "Training dataset 2 (Noisy)": (x2, y2)
    })

    x1, y1 = load_insect_data('insects-testing.txt')
    x2, y2 = load_insect_data('insects-2-testing.txt')

    x1 = x1[60:]  # 只取后 150 个
    y1 = y1[60:]

    plot_insect_comparison({
        "Testing dataset150 1 (Clean)": (x1, y1),
        "Testing dataset150 2 (Noisy)": (x2, y2)
    })

    # 实验 1
    loss1, model1 = train_and_evaluate('insects-training.txt', 'insects-testing.txt')
    
    # 实验 2
    loss2, model2 = train_and_evaluate('insects-2-training.txt', 'insects-2-testing.txt')

    # 一行代码搞定对比
    plot_training_loss(
        {'Clean Data (Exp 1)': loss1, 'Noisy Data (Exp 2)': loss2}, 
        title='Comparison of Training Loss'
    )

    plot_decision_boundary(model1, x1, y1, title="Decision Boundary - Dataset 1 (Clean)")

    plot_decision_boundary(model2, x2, y2, title="Decision Boundary - Dataset 2 (Noisy)")