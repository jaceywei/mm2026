import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib.colors import ListedColormap

# ================= 1. 数据加载与基础绘图 =================

def load_insect_data(file_path):
    try:
        data = np.loadtxt(file_path)
        params = data[:, :2] 
        labels = data[:, 2].astype(int) 
        return params, labels
    except Exception as e:
        print(f"❌ 读取 {file_path} 出错: {e}")
        return None, None

def plot_insect_comparison(dataset_dict):
    n = len(dataset_dict)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1: axes = [axes]
    colors = ['red', 'blue', 'green']
    markers = ['o', 's', '^']
    for ax, (title, (params, labels)) in zip(axes, dataset_dict.items()):
        for i in range(3):
            idx = (labels == i)
            ax.scatter(params[idx, 0], params[idx, 1], c=colors[i], marker=markers[i], alpha=0.6, edgecolors='k')
        ax.set_title(title); ax.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout(); plt.show()

# ================= 2. 灵活的模型定义 =================

class FlexibleInsectNet(nn.Module):
    def __init__(self, hidden_layers=[], activation=nn.ReLU()):
        super(FlexibleInsectNet, self).__init__()
        layers = []
        input_dim = 2
        for h_dim in hidden_layers:
            layers.append(nn.Linear(input_dim, h_dim))
            layers.append(activation)
            input_dim = h_dim
        layers.append(nn.Linear(input_dim, 3)) # 输出 3 类
        self.network = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.network(x)

# ================= 3. 实验运行器 =================

def run_experiment(train_data, test_data, config):
    X_train = torch.FloatTensor(train_data[0]); y_train = torch.LongTensor(train_data[1])
    X_test = torch.FloatTensor(test_data[0]); y_test = torch.LongTensor(test_data[1])
    
    model = FlexibleInsectNet(hidden_layers=config['hidden'], activation=config['act'])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=config['wd'])
    
    loss_hist = []
    for epoch in range(1000):
        model.train()
        optimizer.zero_grad()
        loss = criterion(model(X_train), y_train)
        loss.backward()
        optimizer.step()
        loss_hist.append(loss.item())
        
    model.eval()
    with torch.no_grad():
        # --- 新增：计算训练集准确率 ---
        train_preds = torch.argmax(model(X_train), dim=1)
        acc_train = (train_preds == y_train).float().mean().item()
        
        # 计算测试集准确率
        test_preds = torch.argmax(model(X_test), dim=1)
        acc60 = (test_preds[:60] == y_test[:60]).float().mean().item()
        acc150 = (test_preds[60:] == y_test[60:]).float().mean().item()
        
    return {
        'loss': loss_hist, 
        'model': model, 
        'acc_train': acc_train, # 训练集准确率
        'acc60': acc60, 
        'acc150': acc150
    }

# ================= 4. 对比绘图增强版 =================

def plot_boundaries_separate(exp_results, train_data, test_data, plot_target="Train", title_prefix=""):
    """
    将决策边界排版为 2*2 矩阵，并根据 plot_target 选择只画训练集或只画测试集。
    背景网格范围由全体数据决定，保证两张图的坐标轴和背景色块完全对齐，方便报告对比。
    """
    X_train, y_train = train_data
    X_test, y_test = test_data
    
    # 将训练集和测试集合并，以计算统一的网格范围（极其重要！保证两图背景一致）
    X_all = np.vstack((X_train, X_test))
    x_min, x_max = X_all[:, 0].min() - 0.5, X_all[:, 0].max() + 0.5
    y_min, y_max = X_all[:, 1].min() - 0.5, X_all[:, 1].max() + 0.5
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes_flat = axes.flatten() 
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    grid_tensor = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])
    
    cmap = ListedColormap(['red', 'blue', 'green'])
    colors = ['red', 'blue', 'green']
    
    # 根据目标决定绘制哪部分数据及样式
    if plot_target == "Train":
        X_plot, y_plot = X_train, y_train
        marker, s, edge_c = 'o', 25, 'k'
    else:
        X_plot, y_plot = X_test, y_test
        marker, s, edge_c = '*', 80, 'white' # 测试集用醒目的五角星
        
    for ax, (name, res) in zip(axes_flat, exp_results.items()):
        res['model'].eval()
        with torch.no_grad():
            z = torch.argmax(res['model'](grid_tensor), 1).reshape(xx.shape).numpy()
        
        # 1. 绘制背景色 (决策区域)
        ax.contourf(xx, yy, z, alpha=0.2, cmap=cmap)
        
        # 2. 绘制数据点
        for i in range(3):
            idx = (y_plot == i)
            ax.scatter(X_plot[idx, 0], X_plot[idx, 1], c=colors[i], marker=marker, 
                       edgecolors=edge_c, s=s, alpha=0.8, label=f'Class {i}')
        
        # 3. 设置标题 (训练集标 Train Acc, 测试集标 Test Acc)
        acc_val = res['acc_train'] if plot_target == "Train" else res['acc150']
        ax.set_title(f"{name}\n({plot_target} Acc: {acc_val:.2%})", fontsize=12)
        ax.set_xlabel('Param 1')
        ax.set_ylabel('Param 2')
        ax.grid(True, linestyle='--', alpha=0.3)
    
    # 在第一张子图加图例
    axes_flat[0].legend(loc='upper right', fontsize=10)
    
    plt.suptitle(title_prefix, fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

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

# ================= 5. 主程序：对比实验执行 =================

# ================= 5. 主程序：对比实验执行 =================

if __name__ == "__main__":
    # 加载数据
    data1_train = load_insect_data('insects-training.txt')
    data1_test = load_insect_data('insects-testing.txt')
    data2_train = load_insect_data('insects-2-training.txt')
    data2_test = load_insect_data('insects-2-testing.txt')

    # 定义实验配置
    configs = {
        "Model A (Baseline)": {'hidden': [32, 16], 'act': nn.ReLU(), 'wd': 0},
        "Model B (Linear)":   {'hidden': [], 'act': nn.ReLU(), 'wd': 0},
        "Model C (Complex)":  {'hidden': [64, 64, 64], 'act': nn.ReLU(), 'wd': 0},
        "Model D (Overfit)":  {'hidden': [64, 64, 64, 64], 'act': nn.ReLU(), 'wd': 0},
    }

    num_runs = 10
    
    # 初始化记录历史数据的字典，改为使用列表存储每次 run 的结果
    history_res1 = {name: {'train': [], 'acc60': [], 'acc150': []} for name in configs}
    history_res2 = {name: {'train': [], 'acc60': [], 'acc150': []} for name in configs}

    print(f"🚀 开始执行 {num_runs} 次独立实验并计算平均值与标准差 (已跳过绘图步骤)...")
    
    for i in range(num_runs):
        print(f"\n---> 正在运行第 {i+1}/{num_runs} 次实验...")
        
        # 实验 1 (Dataset 1)
        for name, cfg in configs.items():
            res = run_experiment(data1_train, data1_test, cfg)
            history_res1[name]['train'].append(res['acc_train'])
            history_res1[name]['acc60'].append(res['acc60'])
            history_res1[name]['acc150'].append(res['acc150'])
            
        # 实验 2 (Dataset 2)
        for name, cfg in configs.items():
            res = run_experiment(data2_train, data2_test, cfg)
            history_res2[name]['train'].append(res['acc_train'])
            history_res2[name]['acc60'].append(res['acc60'])
            history_res2[name]['acc150'].append(res['acc150'])


    # --- 打印最终的 10 次平均性能对比表 (包含标准差) ---
    def print_formatted_table(history_dict, dataset_name):
        print("\n" + "="*95)
        print(f"{'Model Name':<20} | {'Train Acc ± Std':<20} | {'Test(60) Acc ± Std':<20} | {'Test(150) Acc ± Std':<20}")
        print("-" * 95)
        print(f"--- {dataset_name} {num_runs}-Run Statistics ---")
        for name in configs.keys():
            res_list = history_dict[name]
            
            # 计算均值和标准差，并转换为百分比格式
            t_mean, t_std = np.mean(res_list['train']) * 100, np.std(res_list['train']) * 100
            a60_mean, a60_std = np.mean(res_list['acc60']) * 100, np.std(res_list['acc60']) * 100
            a150_mean, a150_std = np.mean(res_list['acc150']) * 100, np.std(res_list['acc150']) * 100
            
            # 拼接字符串
            train_str = f"{t_mean:.2f}% ± {t_std:.2f}%"
            acc60_str = f"{a60_mean:.2f}% ± {a60_std:.2f}%"
            acc150_str = f"{a150_mean:.2f}% ± {a150_std:.2f}%"
            
            # 打印对齐格式
            print(f"{name:<20} | {train_str:>18} | {acc60_str:>18} | {acc150_str:>18}")
        print("="*95)

    print_formatted_table(history_res1, "Dataset 1 (Clean)")
    print_formatted_table(history_res2, "Dataset 2 (Noisy)")