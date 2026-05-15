import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# ================= 1. 数据加载 =================

def load_insect_data(file_path):
    try:
        data = np.loadtxt(file_path)
        params = data[:, :2] 
        labels = data[:, 2].astype(int) 
        return params, labels
    except Exception as e:
        print(f"❌ 读取 {file_path} 出错: {e}")
        return None, None

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

# ================= 3. 单次实验运行器 =================

def run_single_experiment(X_train, y_train, X_test, y_test, config):
    """执行单次模型训练并返回准确率"""
    # 注意：为了保证每次运行初始化不同，模型必须在函数内部实例化
    model = FlexibleInsectNet(hidden_layers=config['hidden'], activation=config['act'])
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=config['wd'])
    
    model.train()
    for epoch in range(1000):
        optimizer.zero_grad()
        loss = criterion(model(X_train), y_train)
        loss.backward()
        optimizer.step()
        
    model.eval()
    with torch.no_grad():
        train_preds = torch.argmax(model(X_train), dim=1)
        acc_train = (train_preds == y_train).float().mean().item()
        
        test_preds = torch.argmax(model(X_test), dim=1)
        acc60 = (test_preds[:60] == y_test[:60]).float().mean().item()
        acc150 = (test_preds[60:] == y_test[60:]).float().mean().item()
        
    return acc_train, acc60, acc150

# ================= 4. 多次实验运行与统计 =================

def run_multiple_experiments(train_data, test_data, config, num_runs=10):
    """重复运行多次实验并计算平均值和标准差"""
    X_train = torch.FloatTensor(train_data[0])
    y_train = torch.LongTensor(train_data[1])
    X_test = torch.FloatTensor(test_data[0])
    y_test = torch.LongTensor(test_data[1])
    
    results = {'acc_train': [], 'acc60': [], 'acc150': []}
    
    for _ in range(num_runs):
        acc_train, acc60, acc150 = run_single_experiment(X_train, y_train, X_test, y_test, config)
        results['acc_train'].append(acc_train)
        results['acc60'].append(acc60)
        results['acc150'].append(acc150)
        
    # 计算均值和标准差
    summary = {
        'train_mean': np.mean(results['acc_train']), 'train_std': np.std(results['acc_train']),
        'acc60_mean': np.mean(results['acc60']),     'acc60_std': np.std(results['acc60']),
        'acc150_mean': np.mean(results['acc150']),   'acc150_std': np.std(results['acc150'])
    }
    return summary

def print_results_table(title, results_dict, num_runs):
    """格式化打印实验结果表格"""
    print("\n" + "="*95)
    print(f"{title} (Averaged over {num_runs} runs)")
    print("-" * 95)
    print(f"{'Activation':<15} | {'Train Acc (Mean ± Std)':<22} | {'Test 60 (Mean ± Std)':<22} | {'Test 150 (Mean ± Std)':<22}")
    print("-" * 95)
    
    for name, res in results_dict.items():
        train_str = f"{res['train_mean']:.2%} ± {res['train_std']:.2%}"
        acc60_str = f"{res['acc60_mean']:.2%} ± {res['acc60_std']:.2%}"
        acc150_str = f"{res['acc150_mean']:.2%} ± {res['acc150_std']:.2%}"
        
        print(f"{name:<15} | {train_str:<22} | {acc60_str:<22} | {acc150_str:<22}")
    print("="*95)

# ================= 5. 主程序执行 =================

if __name__ == "__main__":
    # 加载数据
    data1_train = load_insect_data('insects-training.txt')
    data1_test = load_insect_data('insects-testing.txt')
    data2_train = load_insect_data('insects-2-training.txt')
    data2_test = load_insect_data('insects-2-testing.txt')

    # 定义实验配置：固定隐藏层结构 [32, 16]，对比不同的激活函数
    base_hidden = [32, 16]
    configs = {
        "ReLU":      {'hidden': base_hidden, 'act': nn.ReLU(), 'wd': 0},
        "Sigmoid":   {'hidden': base_hidden, 'act': nn.Sigmoid(), 'wd': 0},
        "Tanh":      {'hidden': base_hidden, 'act': nn.Tanh(), 'wd': 0},
        "LeakyReLU": {'hidden': base_hidden, 'act': nn.LeakyReLU(0.1), 'wd': 0},
    }

    NUM_RUNS = 10

    # --- 实验 1: 数据集 1 (Clean) ---
    print(f"🔬 正在执行 Dataset 1 实验 (每种模型独立运行 {NUM_RUNS} 次)...")
    results1 = {}
    for name, cfg in configs.items():
        results1[name] = run_multiple_experiments(data1_train, data1_test, cfg, num_runs=NUM_RUNS)

    # --- 实验 2: 数据集 2 (Noisy) ---
    print(f"🔬 正在执行 Dataset 2 实验 (每种模型独立运行 {NUM_RUNS} 次)...")
    results2 = {}
    for name, cfg in configs.items():
        results2[name] = run_multiple_experiments(data2_train, data2_test, cfg, num_runs=NUM_RUNS)

    # --- 打印统计结果 ---
    print_results_table("--- Dataset 1 (Clean) Results ---", results1, NUM_RUNS)
    print_results_table("--- Dataset 2 (Noisy) Results ---", results2, NUM_RUNS)