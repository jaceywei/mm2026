import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from matplotlib.colors import ListedColormap

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

# ================= 1. 原生 sklearn SVM 实验运行器 =================

def run_sklearn_svm(train_data, test_data, config):
    """直接使用原生 sklearn 的 SVM 进行训练和评估"""
    X_train, y_train = train_data[0], train_data[1]
    X_test, y_test = test_data[0], test_data[1]
    
    # 初始化并训练 SVM
    # C 控制正则化（软间隔），gamma 控制高斯核的带宽
    model = SVC(kernel='rbf', C=config['C'], gamma=config['gamma'])
    model.fit(X_train, y_train)
    
    # 评估准确率
    acc_train = (model.predict(X_train) == y_train).mean()
    test_preds = model.predict(X_test)
    acc60 = (test_preds[:60] == y_test[:60]).mean()
    acc150 = (test_preds[60:] == y_test[60:]).mean()
    
    return {
        'model': model, 
        'acc_train': acc_train, 
        'acc60': acc60, 
        'acc150': acc150
    }

# ================= 2. 专为 SVM 优化的精简版绘图函数 =================

def plot_svm_boundaries(exp_results, train_data, test_data, plot_target="Train", title_prefix=""):
    """专为 sklearn API 修改的绘图函数，去掉了所有 Tensor 操作"""
    X_train, y_train = train_data
    X_test, y_test = test_data
    
    X_all = np.vstack((X_train, X_test))
    x_min, x_max = X_all[:, 0].min() - 0.5, X_all[:, 0].max() + 0.5
    y_min, y_max = X_all[:, 1].min() - 0.5, X_all[:, 1].max() + 0.5
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes_flat = axes.flatten() 
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    # 直接使用 Numpy 数组，无需转换 Tensor
    grid_points = np.c_[xx.ravel(), yy.ravel()] 
    
    cmap = ListedColormap(['red', 'blue', 'green'])
    colors = ['red', 'blue', 'green']
    
    if plot_target == "Train":
        X_plot, y_plot = X_train, y_train
        marker, s, edge_c = 'o', 25, 'k'
    else:
        X_plot, y_plot = X_test, y_test
        marker, s, edge_c = '*', 80, 'white'
        
    for ax, (name, res) in zip(axes_flat, exp_results.items()):
        model = res['model']
        
        # 直接调用 sklearn 的 predict()，极其干净
        z = model.predict(grid_points).reshape(xx.shape)
        
        ax.contourf(xx, yy, z, alpha=0.2, cmap=cmap)
        
        for i in range(3):
            idx = (y_plot == i)
            ax.scatter(X_plot[idx, 0], X_plot[idx, 1], c=colors[i], marker=marker, 
                       edgecolors=edge_c, s=s, alpha=0.8, label=f'Class {i}')
        
        acc_val = res['acc_train'] if plot_target == "Train" else res['acc150']
        ax.set_title(f"{name}\n({plot_target} Acc: {acc_val:.2%})", fontsize=12)
        ax.set_xlabel('Param 1')
        ax.set_ylabel('Param 2')
        ax.grid(True, linestyle='--', alpha=0.3)
    
    axes_flat[0].legend(loc='upper right', fontsize=10)
    plt.suptitle(title_prefix, fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

# ================= 3. 主程序 =================
if __name__ == "__main__":
    # --- 1. 加载所有数据 ---
    # 干净数据集 (Dataset 1)
    data1_train = load_insect_data('insects-training.txt')
    data1_test = load_insect_data('insects-testing.txt')
    
    # 噪声数据集 (Dataset 2)
    data2_train = load_insect_data('insects-2-training.txt')
    data2_test = load_insect_data('insects-2-testing.txt')
    
    # 针对 RBF 核设计的 4 组经典超参数对比
    svm_configs = {
        "SVM (Balanced)":    {'C': 1.0,   'gamma': 'scale'}, # 默认均衡策略
        "SVM (High C)":      {'C': 100.0, 'gamma': 'scale'}, # 强硬划分，容易过拟合（对噪声敏感）
        "SVM (Low C)":       {'C': 0.1,   'gamma': 'scale'}, # 允许更多支持向量，容错率高（抗噪）
        "SVM (High Gamma)":  {'C': 1.0,   'gamma': 5.0},     # 局部影响力过大，边界碎片化
    }

    # --- 2. 运行实验 ---
    results_svm1 = {}
    print("🔬 正在执行 Dataset 1 (干净数据) 的 SVM 实验...")
    for name, cfg in svm_configs.items():
        results_svm1[name] = run_sklearn_svm(data1_train, data1_test, cfg)

    results_svm2 = {}
    print("🔬 正在执行 Dataset 2 (噪声数据) 的 SVM 实验...")
    for name, cfg in svm_configs.items():
        results_svm2[name] = run_sklearn_svm(data2_train, data2_test, cfg)

    # --- 3. 画图对比 ---
    
    # Dataset 1 (Clean) 绘图
    plot_svm_boundaries(
        results_svm1, data1_train, data1_test, 
        plot_target="Train", title_prefix="Dataset 1 (Clean): SVM Boundaries on TRAINING Set"
    )
    plot_svm_boundaries(
        results_svm1, data1_train, data1_test, 
        plot_target="Test", title_prefix="Dataset 1 (Clean): SVM Boundaries on TESTING Set"
    )

    # Dataset 2 (Noisy) 绘图
    plot_svm_boundaries(
        results_svm2, data2_train, data2_test, 
        plot_target="Train", title_prefix="Dataset 2 (Noisy): SVM Boundaries on TRAINING Set"
    )
    plot_svm_boundaries(
        results_svm2, data2_train, data2_test, 
        plot_target="Test", title_prefix="Dataset 2 (Noisy): SVM Boundaries on TESTING Set"
    )

    # --- 4. 打印综合报表 ---
    print("\n" + "="*85)
    print(f"{'SVM Model Name':<20} | {'Train Acc':<12} | {'Test (60)':<12} | {'Test (150)':<12}")
    print("-" * 85)
    
    print("--- Dataset 1 (Clean) Results ---")
    for name, res in results_svm1.items():
        print(f"{name:<20} | {res['acc_train']:>11.2%} | {res['acc60']:>11.2%} | {res['acc150']:>11.2%}")
        
    print("-" * 85)
    print("--- Dataset 2 (Noisy) Results ---")
    for name, res in results_svm2.items():
        print(f"{name:<20} | {res['acc_train']:>11.2%} | {res['acc60']:>11.2%} | {res['acc150']:>11.2%}")
    print("="*85)