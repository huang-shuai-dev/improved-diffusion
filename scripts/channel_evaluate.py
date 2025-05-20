"""
评估生成的信道矩阵与真实信道矩阵的相似度。
"""

import argparse
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from improved_diffusion import logger
from improved_diffusion import dist_util
from improved_diffusion.channel_datasets import ChannelMatrixDataset
from improved_diffusion.channel_datasets import load_data

def evaluate_channel_metrics(real_data, generated_data):
    """
    计算基本误差指标
    
    Args:
        real_data: 真实数据，形状为 [B, 2, H, W]
        generated_data: 生成数据，形状为 [B, 2, H, W]
    
    Returns:
        包含各项误差指标的字典
    """
    # 计算统计特性
    real_mean = torch.mean(real_data, dim=0)  # [2, H, W]
    real_std = torch.std(real_data, dim=0)    # [2, H, W]
    gen_mean = torch.mean(generated_data, dim=0)    # [2, H, W]
    gen_std = torch.std(generated_data, dim=0)      # [2, H, W]
    
    # 打印数据统计信息
    logger.log("\n数据统计信息：")
    logger.log(f"真实数据均值: {torch.mean(real_data).item():.6f}")
    logger.log(f"真实数据标准差: {torch.std(real_data).item():.6f}")
    logger.log(f"真实数据最大值: {torch.max(real_data).item():.6f}")
    logger.log(f"真实数据最小值: {torch.min(real_data).item():.6f}")
    logger.log(f"真实数据均方值: {torch.mean(real_data**2).item():.6f}")
    
    logger.log(f"\n生成数据均值: {torch.mean(generated_data).item():.6f}")
    logger.log(f"生成数据标准差: {torch.std(generated_data).item():.6f}")
    logger.log(f"生成数据最大值: {torch.max(generated_data).item():.6f}")
    logger.log(f"生成数据最小值: {torch.min(generated_data).item():.6f}")
    logger.log(f"生成数据均方值: {torch.mean(generated_data**2).item():.6f}")
    
    # 1. 均值误差
    mean_mse = torch.mean((real_mean - gen_mean)**2).item()
    mean_nmse = (mean_mse / torch.mean(real_mean**2)).item()
    
    # 2. 标准差误差
    std_mse = torch.mean((real_std - gen_std)**2).item()
    std_nmse = (std_mse / torch.mean(real_std**2)).item()
    
    return {
        'mean_mse': mean_mse,
        'mean_nmse': mean_nmse,
        'std_mse': std_mse,
        'std_nmse': std_nmse
    }


def evaluate_statistics(real_data, generated_data):
    """
    评估统计特性
    
    Args:
        real_data: 真实数据，形状为 [B, 2, H, W]
        generated_data: 生成数据，形状为 [B, 2, H, W]
    
    Returns:
        包含统计特性的字典
    """
    # 1. 幅度分布
    real_amplitude = torch.abs(real_data[:,0] + 1j * real_data[:,1])  # [B, H, W]
    gen_amplitude = torch.abs(generated_data[:,0] + 1j * generated_data[:,1])     # [B, H, W]
    
    # 2. 相位分布
    real_phase = torch.angle(real_data[:,0] + 1j * real_data[:,1])    # [B, H, W]
    gen_phase = torch.angle(generated_data[:,0] + 1j * generated_data[:,1])       # [B, H, W]
    
    # 计算统计特性
    stats = {
        'real_amplitude_mean': torch.mean(real_amplitude).item(),
        'gen_amplitude_mean': torch.mean(gen_amplitude).item(),
        'real_amplitude_std': torch.std(real_amplitude).item(),
        'gen_amplitude_std': torch.std(gen_amplitude).item(),
        'real_phase_mean': torch.mean(real_phase).item(),
        'gen_phase_mean': torch.mean(gen_phase).item(),
        'real_phase_std': torch.std(real_phase).item(),
        'gen_phase_std': torch.std(gen_phase).item()
    }
    
    # 计算分布相似度（使用KL散度）
    def compute_kl_divergence(p, q, bins=50):
        # 将数据转移到CPU进行直方图计算
        p = p.cpu().numpy()
        q = q.cpu().numpy()
        # 计算直方图
        p_hist, _ = np.histogram(p.flatten(), bins=bins, density=True)
        q_hist, _ = np.histogram(q.flatten(), bins=bins, density=True)
        # 避免零值
        p_hist = np.clip(p_hist, 1e-10, None)
        q_hist = np.clip(q_hist, 1e-10, None)
        # 计算KL散度
        return np.sum(p_hist * np.log(p_hist / q_hist))
    
    # 计算幅度和相位分布的KL散度
    stats['amplitude_kl_div'] = compute_kl_divergence(real_amplitude, gen_amplitude)
    stats['phase_kl_div'] = compute_kl_divergence(real_phase, gen_phase)
    
    return stats


def evaluate_correlation(real_data, generated_data):
    """
    评估空间相关性
    
    Args:
        real_data: 真实数据，形状为 [B, 2, H, W]
        generated_data: 生成数据，形状为 [B, 2, H, W]
    
    Returns:
        包含相关性指标的字典
    """
    def spatial_correlation(data):
        # 计算空间相关性
        B, C, H, W = data.shape
        corr = torch.zeros((B, H, W), device=data.device)
        
        # 使用向量化操作计算相关性
        for i in range(H):
            for j in range(W):
                if i > 0:
                    # 计算垂直相关性
                    v_corr = torch.corrcoef(torch.stack([
                        data[:, 0, i, j],
                        data[:, 0, i-1, j]
                    ]))[0, 1]
                    corr[:, i, j] += v_corr
                if j > 0:
                    # 计算水平相关性
                    h_corr = torch.corrcoef(torch.stack([
                        data[:, 0, i, j],
                        data[:, 0, i, j-1]
                    ]))[0, 1]
                    corr[:, i, j] += h_corr
        return corr / 2
    
    real_corr = spatial_correlation(real_data)
    gen_corr = spatial_correlation(generated_data)
    
    # 计算相关性统计特性
    real_corr_mean = torch.mean(real_corr, dim=0)  # [H, W]
    real_corr_std = torch.std(real_corr, dim=0)    # [H, W]
    gen_corr_mean = torch.mean(gen_corr, dim=0)    # [H, W]
    gen_corr_std = torch.std(gen_corr, dim=0)      # [H, W]
    
    return {
        'correlation_mean_mse': torch.mean((real_corr_mean - gen_corr_mean)**2).item(),
        'correlation_mean_nmse': (torch.mean((real_corr_mean - gen_corr_mean)**2) / torch.mean(real_corr_mean**2)).item(),
        'correlation_std_mse': torch.mean((real_corr_std - gen_corr_std)**2).item(),
        'correlation_std_nmse': (torch.mean((real_corr_std - gen_corr_std)**2) / torch.mean(real_corr_std**2)).item()
    }


def evaluate_sample_similarity(real_data, generated_data):
    """
    计算每个生成样本与所有真实样本的最小NMSE
    
    Args:
        real_data: 真实数据，形状为 [B_real, 2, H, W]
        generated_data: 生成数据，形状为 [B_gen, 2, H, W]
    
    Returns:
        包含每个生成样本最小NMSE的列表
    """
    B_gen = generated_data.shape[0]
    B_real = real_data.shape[0]
    min_nmse_list = []
    
    logger.log(f"\n计算每个生成样本与真实样本的最小NMSE:")
    logger.log(f"生成样本数量: {B_gen}")
    logger.log(f"真实样本数量: {B_real}")
    
    # 对每个生成样本
    for i in range(B_gen):
        gen_sample = generated_data[i]  # [2, H, W]
        min_nmse = float('inf')
        
        # 与所有真实样本比较
        for j in range(B_real):
            real_sample = real_data[j]  # [2, H, W]
            
            # 计算MSE
            mse = torch.mean((real_sample - gen_sample)**2).item()
            # 计算NMSE
            nmse = mse / torch.mean(real_sample**2).item()
            
            # 更新最小NMSE
            min_nmse = min(min_nmse, nmse)
        
        min_nmse_list.append(min_nmse)
        
        # 每10个样本打印一次进度
        if (i + 1) % 10 == 0:
            logger.log(f"已处理 {i + 1}/{B_gen} 个生成样本")
    
    # 转换为numpy数组以便计算统计信息
    min_nmse_array = np.array(min_nmse_list)
    
    # 打印统计信息
    logger.log("\n最小NMSE统计信息:")
    logger.log(f"最小值: {np.min(min_nmse_array):.6f}")
    logger.log(f"最大值: {np.max(min_nmse_array):.6f}")
    logger.log(f"平均值: {np.mean(min_nmse_array):.6f}")
    logger.log(f"中位数: {np.median(min_nmse_array):.6f}")
    logger.log(f"标准差: {np.std(min_nmse_array):.6f}")
    
    # 打印所有样本的NMSE值
    logger.log("\n所有生成样本的最小NMSE值:")
    for i, nmse in enumerate(min_nmse_list):
        logger.log(f"样本 {i}: {nmse:.6f}")
    
    return min_nmse_list


def visualize_channel_matrix(matrix, title, save_path):
    """
    Visualize channel matrix as image
    
    Args:
        matrix: channel matrix, shape [2, H, W]
        title: image title
        save_path: save path
    """
    # Move data to CPU and convert to numpy array
    matrix = matrix.cpu().numpy()
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot real part
    im1 = ax1.imshow(matrix[0], cmap='viridis')
    ax1.set_title(f'{title} - Real Part')
    plt.colorbar(im1, ax=ax1)
    
    # Plot imaginary part
    im2 = ax2.imshow(matrix[1], cmap='viridis')
    ax2.set_title(f'{title} - Imaginary Part')
    plt.colorbar(im2, ax=ax2)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save image
    plt.savefig(save_path)
    plt.close()


def visualize_comparison(real_data, generated_data, output_dir):
    """
    Visualize comparison between real and generated data
    
    Args:
        real_data: real data, shape [B, 2, H, W]
        generated_data: generated data, shape [B, 2, H, W]
        output_dir: output directory
    """
    # Create visualization directory
    vis_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Select first sample for visualization
    real_sample = real_data[0]
    gen_sample = generated_data[0]
    
    # Visualize real data
    real_path = os.path.join(vis_dir, 'real_channel.png')
    visualize_channel_matrix(real_sample, 'Real Channel Matrix', real_path)
    
    # Visualize generated data
    gen_path = os.path.join(vis_dir, 'generated_channel.png')
    visualize_channel_matrix(gen_sample, 'Generated Channel Matrix', gen_path)
    
    # Visualize multiple samples
    num_samples = min(5, real_data.shape[0], generated_data.shape[0])
    for i in range(num_samples):
        real_sample = real_data[i]
        gen_sample = generated_data[i]
        
        # Visualize real data
        real_path = os.path.join(vis_dir, f'real_channel_{i}.png')
        visualize_channel_matrix(real_sample, f'Real Channel Matrix - Sample {i}', real_path)
        
        # Visualize generated data
        gen_path = os.path.join(vis_dir, f'generated_channel_{i}.png')
        visualize_channel_matrix(gen_sample, f'Generated Channel Matrix - Sample {i}', gen_path)
    
    logger.log(f"Visualization results saved to: {vis_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--real_data", type=str, required=True, help="真实数据文件路径")
    parser.add_argument("--generated_data", type=str, required=True, help="生成数据文件路径")
    parser.add_argument("--output_dir", type=str, default="./evaluation_results", help="评估结果保存目录")
    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    logger.configure(dir=args.output_dir)

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.log(f"使用设备: {device}")

    # 加载真实数据
    dataset = ChannelMatrixDataset(
        file_path=args.real_data,
        pilot_length=8,  # 默认值，可以根据需要修改
        key="output_h",  # 默认值，可以根据需要修改
        snr_db=20.0,     # 默认值，可以根据需要修改
        quantize_y=False, # 评估时不需要量化
        normalize=True   # 保持归一化
    )
    
    # 获取所有真实数据并转移到GPU
    real_data = []
    for i in range(len(dataset)):
        sample = dataset[i]
        real_data.append(sample["H"])
    real_data = torch.stack(real_data).to(device)

    # 加载生成数据并转移到GPU
    generated_data = np.load(args.generated_data)
    if isinstance(generated_data, np.lib.npyio.NpzFile):
        # 如果是npz文件，获取第一个数组
        generated_data = generated_data[generated_data.files[0]]
    if isinstance(generated_data, np.ndarray):
        # 确保数据类型为浮点数
        generated_data = generated_data.astype(np.float32)
        generated_data = torch.from_numpy(generated_data).to(device)
    
    # 打印数据统计信息
    logger.log("\n数据统计信息：")
    logger.log(f"真实数据均值: {torch.mean(real_data).item():.6f}")
    logger.log(f"真实数据标准差: {torch.std(real_data).item():.6f}")
    logger.log(f"真实数据最大值: {torch.max(real_data).item():.6f}")
    logger.log(f"真实数据最小值: {torch.min(real_data).item():.6f}")
    logger.log(f"真实数据均方值: {torch.mean(real_data**2).item():.6f}")
    
    logger.log(f"\n生成数据均值: {torch.mean(generated_data).item():.6f}")
    logger.log(f"生成数据标准差: {torch.std(generated_data).item():.6f}")
    logger.log(f"生成数据最大值: {torch.max(generated_data).item():.6f}")
    logger.log(f"生成数据最小值: {torch.min(generated_data).item():.6f}")
    logger.log(f"生成数据均方值: {torch.mean(generated_data**2).item():.6f}")
    
    logger.log(f"真实数据样本数量: {real_data.shape[0]}")
    logger.log(f"生成数据样本数量: {generated_data.shape[0]}")
    logger.log(f"真实数据类型: {real_data.dtype}")
    logger.log(f"生成数据类型: {generated_data.dtype}")

    # 可视化数据
    visualize_comparison(real_data, generated_data, args.output_dir)

    # 计算每个生成样本的最小NMSE
    min_nmse_list = evaluate_sample_similarity(real_data, generated_data)

    # 计算评估指标
    error_metrics = evaluate_channel_metrics(real_data, generated_data)
    stat_metrics = evaluate_statistics(real_data, generated_data)
    corr_metrics = evaluate_correlation(real_data, generated_data)

    # 打印结果
    logger.log("评估结果：")
    logger.log("\n误差指标：")
    for k, v in error_metrics.items():
        logger.log(f"{k}: {v:.6f}")

    logger.log("\n统计特性：")
    for k, v in stat_metrics.items():
        logger.log(f"{k}: {v:.6f}")

    logger.log("\n相关性指标：")
    for k, v in corr_metrics.items():
        logger.log(f"{k}: {v:.6f}")

    # 保存结果到文件
    results = {
        'error_metrics': error_metrics,
        'stat_metrics': stat_metrics,
        'corr_metrics': corr_metrics,
        'min_nmse_list': min_nmse_list
    }
    np.save(os.path.join(args.output_dir, 'evaluation_results.npy'), results)


if __name__ == "__main__":
    main() 