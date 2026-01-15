# evaluate_metrics.py
import os
import numpy as np
from skimage.measure import compare_psnr, compare_ssim
from skimage import io
from PIL import Image
import torch
from torchvision import transforms

def calculate_psnr_ssim(gt_path, pred_path):
    """计算单张图像的PSNR和SSIM"""
    # 读取图像
    gt = io.imread(gt_path)
    pred = io.imread(pred_path)
    
    # 确保图像范围在[0, 255]
    if gt.max() <= 1.0:
        gt = (gt * 255).astype(np.uint8)
    if pred.max() <= 1.0:
        pred = (pred * 255).astype(np.uint8)
    
    # 计算PSNR和SSIM
    psnr = compare_psnr(gt, pred)
    ssim = compare_ssim(gt, pred, multichannel=True, channel_axis=2)
    
    return psnr, ssim

def evaluate_all_images(gt_dir, pred_dir, output_file='metrics_results.txt'):
    """
    评估所有图像的指标
    gt_dir: Ground Truth图像目录 (datasets/dunhuang/test_ref)
    pred_dir: 模型生成的图像目录 (experiments/xxx/results/test/xxx/)
    """
    # 获取所有GT图像
    gt_files = sorted([f for f in os.listdir(gt_dir) if f.endswith(('.png', '.jpg', '.JPG', '.PNG'))])
    
    all_psnr = []
    all_ssim = []
    results = []
    
    print(f"找到 {len(gt_files)} 张GT图像")
    print(f"开始评估...")
    
    for gt_file in gt_files:
        gt_path = os.path.join(gt_dir, gt_file)
        
        # 查找对应的预测图像
        # 预测图像文件名格式可能是: Out_xxx.png 或直接是 xxx.png
        pred_file = None
        
        # 尝试多种可能的文件名格式
        possible_names = [
            f"Out_{gt_file}",
            gt_file,
            gt_file.replace('.png', '.png'),
            gt_file.replace('.jpg', '.png'),
        ]
        
        for name in possible_names:
            pred_path = os.path.join(pred_dir, name)
            if os.path.exists(pred_path):
                pred_file = name
                break
        
        if pred_file is None:
            print(f"警告: 未找到 {gt_file} 对应的预测图像，跳过")
            continue
        
        pred_path = os.path.join(pred_dir, pred_file)
        
        try:
            psnr, ssim = calculate_psnr_ssim(gt_path, pred_path)
            all_psnr.append(psnr)
            all_ssim.append(ssim)
            results.append((gt_file, psnr, ssim))
            print(f"{gt_file}: PSNR={psnr:.4f}, SSIM={ssim:.4f}")
        except Exception as e:
            print(f"处理 {gt_file} 时出错: {e}")
            continue
    
    # 计算平均值
    if len(all_psnr) > 0:
        avg_psnr = np.mean(all_psnr)
        avg_ssim = np.mean(all_ssim)
        std_psnr = np.std(all_psnr)
        std_ssim = np.std(all_ssim)
        
        print("\n" + "="*50)
        print("评估结果汇总:")
        print(f"平均 PSNR: {avg_psnr:.4f} ± {std_psnr:.4f} dB")
        print(f"平均 SSIM: {avg_ssim:.4f} ± {std_ssim:.4f}")
        print(f"评估图像数量: {len(all_psnr)}")
        print("="*50)
        
        # 保存结果到文件
        with open(output_file, 'w') as f:
            f.write("图像评估结果\n")
            f.write("="*50 + "\n")
            for img_name, psnr, ssim in results:
                f.write(f"{img_name}: PSNR={psnr:.4f}, SSIM={ssim:.4f}\n")
            f.write("\n" + "="*50 + "\n")
            f.write(f"平均 PSNR: {avg_psnr:.4f} ± {std_psnr:.4f} dB\n")
            f.write(f"平均 SSIM: {avg_ssim:.4f} ± {std_ssim:.4f}\n")
            f.write(f"评估图像数量: {len(all_psnr)}\n")
        
        print(f"\n结果已保存到: {output_file}")
        
        return avg_psnr, avg_ssim
    else:
        print("错误: 没有成功评估任何图像")
        return None, None

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='评估模型生成的图像质量')
    parser.add_argument('--gt_dir', type=str, required=True, 
                       help='Ground Truth图像目录 (例如: datasets/dunhuang/test_ref)')
    parser.add_argument('--pred_dir', type=str, required=True,
                       help='模型生成的图像目录 (例如: experiments/test_dunhuang_xxx/results/test/xxx/)')
    parser.add_argument('--output', type=str, default='metrics_results.txt',
                       help='输出结果文件路径')
    
    args = parser.parse_args()
    
    evaluate_all_images(args.gt_dir, args.pred_dir, args.output)