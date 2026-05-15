import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from matplotlib.colors import ListedColormap
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

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
def run_optimized_svm(train_data, test_data):
    X_train, y_train = train_data
    X_test, y_test = test_data

    # 1. 创建流水线：先标准化，再跑 SVM
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(kernel='rbf'))
    ])

    # 2. 定义网格搜索范围
    param_grid = {
        'svm__C': [0.1, 1, 10, 100, 10000],
        'svm__gamma': [0.001, 0.01, 0.1, 1, 10]
    }

    # 3. 自动寻找在该数据集上的最优组合
    grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    
    # 打印最优参数，帮你分析为什么之前的配置不好
    print(f"✅ 最优参数: {grid_search.best_params_}")

    # 评估
    acc_train = best_model.score(X_train, y_train)
    test_preds = best_model.predict(X_test)
    acc60 = (test_preds[:60] == y_test[:60]).mean()
    acc150 = (test_preds[60:] == y_test[60:]).mean()

    return {
        'model': best_model,
        'acc_train': acc_train,
        'acc60': acc60,
        'acc150': acc150
    }

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
def plot_best_svm_boundary(best_model, train_data, test_data, dataset_name="Dataset"):
    """
    专门为单一最优模型绘制 1x2 的对比图：左侧为训练集，右侧为测试集。
    """
    X_train, y_train = train_data
    X_test, y_test = test_data
    
    # 将训练集和测试集合并计算统一的网格范围，确保左右两图背景完全一致
    X_all = np.vstack((X_train, X_test))
    x_min, x_max = X_all[:, 0].min() - 0.5, X_all[:, 0].max() + 0.5
    y_min, y_max = X_all[:, 1].min() - 0.5, X_all[:, 1].max() + 0.5
    
    # 创建 1x2 的画布
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    grid_points = np.c_[xx.ravel(), yy.ravel()] 
    
    # 预测网格点 (最优模型包含 StandardScaler，会自动处理缩放)
    z = best_model.predict(grid_points).reshape(xx.shape)
    
    cmap = ListedColormap(['red', 'blue', 'green'])
    colors = ['red', 'blue', 'green']
    
    # 定义左右两图的配置字典：[坐标轴, 数据X, 数据Y, 形状, 大小, 边缘颜色, 标题]
    plot_configs = [
        (axes[0], X_train, y_train, 'o', 30, 'k', f"{dataset_name}: Training Set Boundary"),
        (axes[1], X_test, y_test, '*', 100, 'white', f"{dataset_name}: Testing Set Boundary")
    ]
    
    for ax, X_plot, y_plot, marker, s, edge_c, title in plot_configs:
        # 画决策区域
        ax.contourf(xx, yy, z, alpha=0.2, cmap=cmap)
        
        # 画散点
        for i in range(3):
            idx = (y_plot == i)
            ax.scatter(X_plot[idx, 0], X_plot[idx, 1], c=colors[i], marker=marker, 
                       edgecolors=edge_c, s=s, alpha=0.8, label=f'Class {i}')
        
        ax.set_title(title, fontsize=13)
        ax.set_xlabel('Param 1')
        ax.set_ylabel('Param 2')
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend(loc='upper right', fontsize=10)
        
    plt.tight_layout()
    plt.show()

# ================= 3. 主程序 =================
if __name__ == "__main__":
    # --- 1. 加载数据 ---
    data1_train = load_insect_data('insects-training.txt')
    data1_test = load_insect_data('insects-testing.txt')
    data2_train = load_insect_data('insects-2-training.txt')
    data2_test = load_insect_data('insects-2-testing.txt')

    # --- 2. 针对 Dataset 1 (Clean) 搜索、绘图与输出 ---
    print("\n" + "="*50)
    print("🔍 正在通过网格搜索优化 Dataset 1 (Clean)...")
    res_opt1 = run_optimized_svm(data1_train, data1_test)
    
    # 绘制 1x2 图
    plot_best_svm_boundary(res_opt1['model'], data1_train, data1_test, dataset_name="Dataset 1 (Clean)")
    
    print("\n✅ Dataset 1 最佳参数组合:")
    print(f"C: {res_opt1['model'].named_steps['svm'].C}")
    print(f"Gamma: {res_opt1['model'].named_steps['svm'].gamma}")
    print("-" * 50)
    print(f"{'Metric':<15} | {'Accuracy':<10}")
    print(f"{'Train Acc':<15} | {res_opt1['acc_train']:.2%}")
    print(f"{'Test (60) Acc':<15} | {res_opt1['acc60']:.2%}")
    print(f"{'Test (150) Acc':<15} | {res_opt1['acc150']:.2%}")
    print("="*50)

    # --- 3. 针对 Dataset 2 (Noisy) 搜索、绘图与输出 ---
    print("\n\n" + "="*50)
    print("🔍 正在通过网格搜索优化 Dataset 2 (Noisy)...")
    res_opt2 = run_optimized_svm(data2_train, data2_test)
    
    # 绘制 1x2 图
    plot_best_svm_boundary(res_opt2['model'], data2_train, data2_test, dataset_name="Dataset 2 (Noisy)")
    
    print("\n✅ Dataset 2 最佳参数组合:")
    print(f"C: {res_opt2['model'].named_steps['svm'].C}")
    print(f"Gamma: {res_opt2['model'].named_steps['svm'].gamma}")
    print("-" * 50)
    print(f"{'Metric':<15} | {'Accuracy':<10}")
    print(f"{'Train Acc':<15} | {res_opt2['acc_train']:.2%}")
    print(f"{'Test (60) Acc':<15} | {res_opt2['acc60']:.2%}")
    print(f"{'Test (150) Acc':<15} | {res_opt2['acc150']:.2%}")
    print("="*50)