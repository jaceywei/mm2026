import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# ================= 1. 数据加载 (完全复用您的逻辑) =================

def load_insect_data(file_path):
    try:
        data = np.loadtxt(file_path)
        params = data[:, :2] 
        labels = data[:, 2].astype(int) 
        return params, labels
    except Exception as e:
        print(f"❌ 读取 {file_path} 出错: {e}")
        return None, None

# ================= 2. KNN 实验运行器 =================

def run_knn_experiment(train_data, test_data, k_neighbors):
    X_train, y_train = train_data
    X_test, y_test = test_data
    knn = KNeighborsClassifier(n_neighbors=k_neighbors)
    knn.fit(X_train, y_train)
    
    train_preds = knn.predict(X_train)
    acc_train = accuracy_score(y_train, train_preds)
    test_preds = knn.predict(X_test)
    acc60 = accuracy_score(y_test[:60], test_preds[:60]) if len(y_test) >= 60 else 0
    acc150 = accuracy_score(y_test[60:], test_preds[60:]) if len(y_test) > 60 else 0
    
    return {
        'model': knn, 
        'acc_train': acc_train, 
        'acc60': acc60, 
        'acc150': acc150,
        'acc_test': accuracy_score(y_test, test_preds)
    }

# ================= 3. 高精度清晰边界版绘图 =================

def plot_knn_boundaries_high_res(exp_results, train_data, test_data, plot_target="Train", title_prefix=""):
    X_train, y_train = train_data
    X_test, y_test = test_data
    
    X_all = np.vstack((X_train, X_test))
    x_min, x_max = X_all[:, 0].min() - 0.5, X_all[:, 0].max() + 0.5
    y_min, y_max = X_all[:, 1].min() - 0.5, X_all[:, 1].max() + 0.5
    
    n = len(exp_results)
    rows = int(np.ceil(n / 2))
    cols = 2 if n > 1 else 1
    fig, axes = plt.subplots(rows, cols, figsize=(14, 6 * rows))
    if n == 1: axes = np.array([axes])
    axes_flat = axes.flatten() 
    
    res_step = 0.02
    xx, yy = np.meshgrid(np.arange(x_min, x_max, res_step), np.arange(y_min, y_max, res_step))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    # 颜色定义
    c_red   = np.array([1.0, 0.7, 0.7])  
    c_blue  = np.array([0.7, 0.7, 1.0])  
    c_green = np.array([0.7, 1.0, 0.7])  
    colors_scatter = ['red', 'blue', 'green']
    
    if plot_target == "Train":
        X_plot, y_plot = X_train, y_train
        marker, s, edge_c = 'o', 30, 'k'
    else:
        X_plot, y_plot = X_test, y_test
        marker, s, edge_c = '*', 100, 'white'
        
    print(f"🎨 正在绘制 {title_prefix} (精度: {res_step})...")

    for ax, (name, res) in zip(axes_flat, exp_results.items()):
        knn = res['model']
        
        # 严格计算概率
        probas = knn.predict_proba(grid_points)
        probas_full = np.zeros((len(grid_points), 3))
        for idx, cls in enumerate(knn.classes_):
            probas_full[:, cls] = probas[:, idx]
            
        # 严格平局逻辑：必须概率完全相等
        max_p = np.max(probas_full, axis=1, keepdims=True)
        # 考虑到浮点数精度，使用微小的 epsilon 比对
        is_max = (np.abs(probas_full - max_p) < 1e-9)
        
        # 归一化权重 (当并列第一时，权重被平分，产生混合色)
        weights = is_max.astype(float) / np.sum(is_max, axis=1, keepdims=True)
            
        grid_colors = (weights[:, 0:1] * c_red + 
                       weights[:, 1:2] * c_blue + 
                       weights[:, 2:3] * c_green)
        grid_colors = grid_colors.reshape(xx.shape[0], xx.shape[1], 3)
        
        ax.imshow(grid_colors, extent=(x_min, x_max, y_min, y_max), 
                  origin='lower', aspect='auto', interpolation='nearest', alpha=1.0)
        
        for i in range(3):
            idx = (y_plot == i)
            ax.scatter(X_plot[idx, 0], X_plot[idx, 1], c=colors_scatter[i], marker=marker, 
                       edgecolors=edge_c, s=s, alpha=1.0, label=f'Class {i}')
        
        acc_val = res['acc_train'] if plot_target == "Train" else res['acc150']
        ax.set_title(f"{name}\n({plot_target} Acc: {acc_val:.2%})", fontsize=13)
        ax.grid(False) # 高精度下关闭网格线，让边界更纯净

    for i in range(n, len(axes_flat)): fig.delaxes(axes_flat[i])
    axes_flat[0].legend(loc='upper right')
    plt.suptitle(title_prefix, fontsize=18, y=1.02)
    plt.tight_layout()
    plt.show()

# ================= 4. 执行流程 =================

if __name__ == "__main__":
    # 加载数据 (请确保文件存在)
    data1_train = load_insect_data('insects-training.txt')
    data1_test = load_insect_data('insects-testing.txt')
    data2_train = load_insect_data('insects-2-training.txt')
    data2_test = load_insect_data('insects-2-testing.txt')

    knn_configs = {"KNN (k=1)": 1, "KNN (k=3)": 3, "KNN (k=5)": 5, "KNN (k=15)": 15}
    
    # 运行实验
    res1 = {n: run_knn_experiment(data1_train, data1_test, k) for n, k in knn_configs.items()}
    res2 = {n: run_knn_experiment(data2_train, data2_test, k) for n, k in knn_configs.items()}

    # --- 输出 4 幅高精度图 ---
    plot_knn_boundaries_high_res(res1, data1_train, data1_test, "Train", "Dataset 1: Training Set")
    plot_knn_boundaries_high_res(res1, data1_train, data1_test, "Test",  "Dataset 1: Testing Set")
    plot_knn_boundaries_high_res(res2, data2_train, data2_test, "Train", "Dataset 2: Training Set")
    plot_knn_boundaries_high_res(res2, data2_train, data2_test, "Test",  "Dataset 2: Testing Set")

    # --- 打印对比表 ---
    print("\n" + "="*85)
    print(f"{'KNN Model (K)':<25} | {'Train Acc':<12} | {'Test (60)':<12} | {'Test (150)':<12}")
    print("-" * 85)
    for title, results in [("Dataset 1 (Clean)", res1), ("Dataset 2 (Noisy)", res2)]:
        print(f"--- {title} ---")
        for name, r in results.items():
            print(f"{name:<25} | {r['acc_train']:>11.2%} | {r['acc60']:>11.2%} | {r['acc150']:>11.2%}")
        print("-" * 85)
    print("="*85)