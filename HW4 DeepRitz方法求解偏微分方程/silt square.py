import torch
from typing import Callable
import torch.nn as nn
import math
import numpy as np
import matplotlib.pyplot as plt

def varphi(x):
    # 自定义激活函数（不变）
    x_cubed = torch.pow(x, 3)
    return torch.max(x_cubed, torch.tensor(0.0, device=x.device))

class FirstResBlock(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, activation: Callable = varphi):
        super(FirstResBlock, self).__init__()
        self.activation = activation
        # 主路径：第一层负责将维度从 2 提升到 10
        self.linear1 = nn.Linear(input_dim, hidden_dim) # 30 参数
        self.linear2 = nn.Linear(hidden_dim, hidden_dim) # 110 参数
        
        # Shortcut 路径：论文中使用补零，0 参数
        self.padding_size = hidden_dim - input_dim

    def forward(self, s):
        # 主路径
        out = self.activation(self.linear1(s))
        out = self.activation(self.linear2(out))
        
        # Shortcut 路径：对原始输入 s (batch, 2) 在最后一个维度补 8 个 0 变为 (batch, 10)
        # F.pad 的参数 (0, 8) 表示在维度最后补 8 个 0
        import torch.nn.functional as F
        identity = F.pad(s, (0, self.padding_size), "constant", 0)
        
        return out + identity

# 【核心重写2】后续块：无维度适配（仅残差运算，hidden_dim→hidden_dim）
class SubsequentResBlock(nn.Module):
    def __init__(self, hidden_dim: int, activation: Callable = varphi):
        super(SubsequentResBlock, self).__init__()
        self.hidden_dim = hidden_dim
        self.activation = activation

        # 无维度适配层（文章要求：仅残差运算）
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, s):
        # 直接用hidden_dim维输入做残差运算（无需适配）
        out = self.activation(self.linear1(s))
        out = self.activation(self.linear2(out))
        return out + s

# 【重写3】网络构建函数：按“第一个块 + 后续块 + 输出层”堆叠
def build_multi_layer_resnet(
    input_dim: int,
    num_layers: int,
    hidden_dim: int,
    activation: Callable = varphi,
    output_dim: int = 1
) -> nn.Sequential:
    """
    严格匹配文章结构：
    输入 → 第一个块（补0/线性层适配→hidden_dim）→ 后续N-1个块（仅残差）→ 输出层（hidden_dim→input_dim）
    """
    assert num_layers >= 1, "num_layers至少为1（必须有一个第一个块）"
    
    all_layers = []
    
    # 1. 加入第一个块（维度适配+残差）
    all_layers.append(FirstResBlock(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        activation=activation
    ))
    
    # 2. 加入后续N-1个块（无适配，仅残差）
    if num_layers > 1:
        all_layers.extend([
            SubsequentResBlock(
                hidden_dim=hidden_dim,
                activation=activation
            ) for _ in range(num_layers - 1)
        ])
    
    # 3. 输出层：hidden_dim → output_dim（还原输入维度）
    all_layers.append(nn.Linear(hidden_dim, output_dim))

    # print(all_layers)
    return nn.Sequential(*all_layers)

def compute_functional_loss(
    model: nn.Module, # 神经网络模型u(x;θ)
    x: torch.Tensor,  # 随机采样的域内点（batch_size × input_dim）
    f: Callable[[torch.Tensor], torch.Tensor],  # 外部强迫函数f(x)
    g: Callable[[torch.Tensor], torch.Tensor],  # 边界函数g(x)
    beta: float = 0.0,  # 边界惩罚系数（仅当处理Dirichlet边界时非零，对应论文公式(3.2)）
    x_boundary: torch.Tensor = None  # 随机采样的边界点（batch_size × input_dim，可选）
) -> torch.Tensor:
    """
    计算变分问题的目标泛函L(θ)（论文公式(2.8)）：
    L(θ) = ∫_Ω [ (1/2)|∇u(x;θ)|² - f(x)u(x;θ) ] dx + β∫_∂Ω u(x;θ)² ds（边界惩罚项）
    """
    # 启用梯度计算（需计算u对x的偏导，即∇u）
    x.requires_grad_(True)
    
    # 1. 前向传播：获取模型输出u(x;θ)（shape: batch_size × 1，模型应直接输出标量解）
    u = model(x)  # 若模型输出是(batch_size,)，这里改为 model(x).unsqueeze(1)；若已是(batch_size,1)，直接用model(x)
    # （根据你的模型定义，二选一：如果build_multi_layer_resnet输出是1维，保留unsqueeze；否则直接用u=model(x)）
    # 这里按论文标量解调整：确保u是(batch_size,1)
    if u.dim() == 1:
        u = u.unsqueeze(1)
    
    # 2. 计算梯度∇u（论文公式(2.7)中的|∇u|²项）
    grad_u = torch.autograd.grad(
        outputs=u.sum(),  # 对u求和以得到标量，便于计算梯度
        inputs=x,
        create_graph=True,
        retain_graph=True
    )[0]  # grad_u shape: batch_size × input_dim（每个点的梯度向量）

    # 3. 域内积分项（论文公式(2.7)的均值近似）
    term_domain = 0.5 * torch.norm(grad_u, p=2, dim=1).square() - f(x) * u.squeeze(1)
    loss_domain = term_domain.mean()
    
    # 4. 边界惩罚项（修复2个问题：拼写错误squuze→squeeze + 冗余unsqueeze）
    loss_boundary = torch.tensor(0.0, device=x.device)
    if beta > 0 and x_boundary is not None:
        # 修复1：squuze→squeeze（致命拼写错误！）
        # 修复2：模型输出若已是(batch_size,1)，直接用；若为(batch_size,)，squeeze后无需再unsqueeze（后续减g_boundary会自动广播）
        u_boundary = model(x_boundary)
        if u_boundary.dim() == 1:
            u_boundary = u_boundary.unsqueeze(1)
        g_boundary = g(x_boundary).unsqueeze(1)  # 确保g输出是(batch_size,1)
        loss_boundary = beta * (u_boundary - g_boundary).square().mean()  # 去掉多余空格
    
    # 总泛函损失
    total_loss = loss_domain + loss_boundary
    
    # 关闭x的梯度计算
    x.requires_grad_(False)
    
    return total_loss

# ------------------------------------------------------------------------------
# 1. 定义算例的精确解与边界函数
# ------------------------------------------------------------------------------
def get_r_theta(tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x = tensor[:, 0]
    y = tensor[:, 1]
    # 计算极径 r
    r = torch.sqrt(x**2 + y**2)
    # 计算极角 theta，并将其值域从 (-pi, pi] 映射到 [0, 2pi)
    theta = torch.atan2(y, x)
    theta = torch.where(theta < 0, theta + 2 * math.pi, theta)
    return r, theta

def exact_solution_slit(tensor: torch.Tensor) -> torch.Tensor:
    r, theta = get_r_theta(tensor)
    # u(r, theta) = r^(1/2) * sin(theta / 2)
    return torch.sqrt(r) * torch.sin(theta / 2.0)

def f_slit(tensor: torch.Tensor) -> torch.Tensor:
    # 源项 -Δu = 0
    return torch.zeros(tensor.shape[0], device=tensor.device)

def g_slit(tensor: torch.Tensor) -> torch.Tensor:
    # Dirichlet 边界条件即为精确解在边界上的取值
    return exact_solution_slit(tensor)

# ------------------------------------------------------------------------------
# 2. 针对裂缝区域 (Slit Domain) 的自定义采样函数
# ------------------------------------------------------------------------------
def sample_random_points_slit(
    input_dim: int,
    batch_size: int,
    domain: tuple[float, float] = (-1.0, 1.0),
    boundary_ratio: float = 0.2
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    专门针对带有 [0, 1) x {0} 裂缝的区域进行采样。
    将边界点均匀分配给 4 条外边 和 1 条裂缝。
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 域内采样 Ω = (-1,1)×(-1,1) \ [0,1)×{0}
    # 裂缝的面积测度为0，因此直接在正方形内随机采样即可作为域内点
    x_domain = torch.rand(batch_size, input_dim, device=device)
    x_domain = x_domain * (domain[1] - domain[0]) + domain[0]
    
    # 2. 边界采样 ∂Ω (包含4条外边和1条内部裂缝)
    boundary_batch_size = int(batch_size * boundary_ratio)
    
    # 分成 5 个部分（4条边 + 1个裂缝）
    num_per_part = boundary_batch_size // 5
    remainder = boundary_batch_size - 4 * num_per_part
    
    x_boundary_list = []
    
    # (1) 上边 y = 1
    x_top = torch.rand(num_per_part, 2, device=device) * 2 - 1
    x_top[:, 1] = 1.0
    x_boundary_list.append(x_top)
    
    # (2) 下边 y = -1
    x_bottom = torch.rand(num_per_part, 2, device=device) * 2 - 1
    x_bottom[:, 1] = -1.0
    x_boundary_list.append(x_bottom)
    
    # (3) 左边 x = -1
    x_left = torch.rand(num_per_part, 2, device=device) * 2 - 1
    x_left[:, 0] = -1.0
    x_boundary_list.append(x_left)
    
    # (4) 右边 x = 1
    x_right = torch.rand(num_per_part, 2, device=device) * 2 - 1
    x_right[:, 0] = 1.0
    x_boundary_list.append(x_right)
    
    # (5) 内部裂缝 x ∈ [0, 1), y = 0
    x_slit = torch.zeros(remainder, 2, device=device)
    x_slit[:, 0] = torch.rand(remainder, device=device)  # x 在 [0, 1) 之间
    # y 保持为 0
    x_boundary_list.append(x_slit)
    
    x_boundary = torch.cat(x_boundary_list, dim=0)
    
    return x_domain, x_boundary

# ------------------------------------------------------------------------------
# 3. 定制化的训练包装函数 (替换其内部调用的 sampler)
# ------------------------------------------------------------------------------
def train_deep_ritz_slit(
    input_dim=2, num_layers=4, hidden_dim=10, 
    num_iterations=25000, lr=1e-3, beta=500.0, batch_size=1000
):
    print("Number of layers:", num_layers)
    print("Number of parameters in the model:", sum(p.numel() for p in build_multi_layer_resnet(input_dim, num_layers, hidden_dim).parameters()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training Slit Domain on device: {device}")
    
    model = build_multi_layer_resnet(
        input_dim=input_dim, num_layers=num_layers, hidden_dim=hidden_dim
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_history = []
    
    model.train()
    for iter_idx in range(num_iterations):
        # !! 这里使用针对裂缝特化的采样函数 !!
        x_domain, x_boundary = sample_random_points_slit(
            input_dim=input_dim, batch_size=batch_size, domain=(-1.0, 1.0)
        )
        
        optimizer.zero_grad()
        
        # 复用你的泛函计算模块
        loss = compute_functional_loss(
            model=model, x=x_domain, f=f_slit, g=g_slit, beta=beta, x_boundary=x_boundary
        )
        
        loss.backward()
        optimizer.step()
        
        loss_history.append(loss.item())
        if (iter_idx + 1) % 5000 == 0:
            avg_loss = sum(loss_history[-5000:]) / 5000
            print(f"Iteration [{iter_idx + 1}/{num_iterations}] | Avg Loss: {avg_loss:.6f}")
            
    return model, loss_history

def plot_error_distribution(model, exact_func, x_range=(-1.0, 1.0), y_range=(-1.0, 1.0), grid_size=200):
    """
    绘制模型预测值、解析解以及绝对误差在正方形区域上的分布图。
    """
    device = next(model.parameters()).device
    model.eval()

    # 1. 生成网格点
    x = np.linspace(x_range[0], x_range[1], grid_size)
    y = np.linspace(y_range[0], y_range[1], grid_size)
    X, Y = np.meshgrid(x, y)
    
    # 展平并转为 tensor
    xy_flat = np.stack([X.ravel(), Y.ravel()], axis=1)
    xy_tensor = torch.tensor(xy_flat, dtype=torch.float32, device=device)

    # 2. 计算预测值和解析解
    with torch.no_grad():
        u_pred_flat = model(xy_tensor).cpu().numpy().flatten()
        u_exact_flat = exact_func(xy_tensor).cpu().numpy().flatten()
    
    # 计算绝对误差
    error_flat = np.abs(u_pred_flat - u_exact_flat)

    # 重新整理形状
    U_pred = u_pred_flat.reshape(grid_size, grid_size)
    U_exact = u_exact_flat.reshape(grid_size, grid_size)
    Error = error_flat.reshape(grid_size, grid_size)

    # 3. 开始绘图
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 子图1：解析解
    im0 = axes[0].pcolormesh(X, Y, U_exact, cmap='jet', shading='auto')
    axes[0].set_title("Exact Solution $u^*$")
    axes[0].set_aspect('equal')
    fig.colorbar(im0, ax=axes[0])

    # 子图2：模型预测
    im1 = axes[1].pcolormesh(X, Y, U_pred, cmap='jet', shading='auto')
    axes[1].set_title("Deep Ritz Prediction $u_\\theta$")
    axes[1].set_aspect('equal')
    fig.colorbar(im1, ax=axes[1])

    # 子图3：绝对误差分布 (Log Scale 更有利于观察误差分布)
    # 如果误差包含0，可以用 abs_error + 1e-10
    im2 = axes[2].pcolormesh(X, Y, Error, cmap='viridis', shading='auto')
    axes[2].set_title("Absolute Error $|u_\\theta - u^*|$")
    axes[2].set_aspect('equal')
    fig.colorbar(im2, ax=axes[2])

    for ax in axes:
        ax.set_xlabel('x')
        ax.set_ylabel('y')

    plt.tight_layout()
    plt.show()

def plot_loss_curve(loss_history, title="Deep Ritz Energy Functional Convergence"):
    """
    绘制训练步数与能量泛函损失的对应关系图（线性坐标系）
    """
    plt.figure(figsize=(10, 5))
    
    # 1. 绘制原始损失（浅色，展示随机采样的波动性）
    iterations = np.arange(len(loss_history))
    plt.plot(iterations, loss_history, color='#1f77b4', alpha=0.3, label='Raw Loss (Stochastic)')
    
    # 2. 绘制滑动平均线（深色，展示整体收敛趋势）
    # 窗口大小可以根据总迭代次数调整，这里取 100
    if len(loss_history) > 100:
        window_size = 100
        smooth_loss = np.convolve(loss_history, np.ones(window_size)/window_size, mode='valid')
        plt.plot(np.arange(window_size-1, len(loss_history)), smooth_loss, 
                 color='#d62728', linewidth=2, label=f'Moving Average (window={window_size})')

    plt.title(title, fontsize=14)
    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel('Energy Functional Value $J[u_\\theta]$', fontsize=12)
    
    # 添加一条水平基准线 y=0 (如果 loss 跨越了正负)
    plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
    
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------------------------
# 4. 执行训练并计算误差
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    # 执行训练
    num_layers = 5
    model_slit, loss_hist = train_deep_ritz_slit(
        num_layers=num_layers, hidden_dim=10, num_iterations=50000
    )
    
    # --- 误差评估 ---
    model_slit.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 生成测试网格（避开原点奇异性）
    x = np.linspace(-1, 1, 100)
    y = np.linspace(-1, 1, 100)
    X, Y = np.meshgrid(x, y)
    
    # 转换为 tensor 进行预测
    test_pts = np.vstack((X.flatten(), Y.flatten())).T
    test_tensor = torch.tensor(test_pts, dtype=torch.float32, device=device)
    
    with torch.no_grad():
        u_pred = model_slit(test_tensor).cpu().numpy().flatten()
        u_exact = exact_solution_slit(test_tensor).cpu().numpy().flatten()
        
    # 计算相对 L2 误差并输出误差分布图
    l2_error = np.linalg.norm(u_pred - u_exact) / np.linalg.norm(u_exact)
    print(f"Relative L2 Error for Slit Domain: {l2_error:.4e}")
    plot_error_distribution(model_slit, exact_solution_slit)
    # plot_loss_curve(loss_hist)