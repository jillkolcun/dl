# -*- coding: utf-8 -*-
"""
表面形貌粗糙度评定 Demo —— 针对「基于深度学习的超精密加工表面测量与评定」

功能：
  1. 检查 DL 环境（torch + CUDA 是否可用）
  2. 读取一张表面形貌图（TIFF/PNG/...），或自动生成一张仿真加工表面
  3. 计算 ISO 25178 面粗糙度参数：Sa / Sq / Sz
  4. 画出 3D 表面图，并保存为 PNG
  5. （可选）把结果写入 Excel 报表

运行方式（在 dl1 环境里）：
  python surface_roughness_demo.py                 # 用自动生成的仿真表面
  python surface_roughness_demo.py surface.tif     # 用你自己的表面图
"""

import sys
import numpy as np

# ---------- 0. 环境检查（确认你的 dl1 深度学习环境正常）----------
print("=" * 60)
print("0. 深度学习环境检查")
print("=" * 60)
try:
    import torch
    if torch.cuda.is_available():
        print(f"  torch {torch.__version__}  ✅ CUDA 可用: {torch.cuda.get_device_name(0)}")
    else:
        print(f"  torch {torch.__version__}  ⚠️ CUDA 不可用（将用 CPU）")
except Exception as e:
    print(f"  torch 未安装或导入失败: {e}")


# ---------- 1. 载入 / 生成表面数据 ----------
def load_or_synth(path=None):
    """载入真实表面图；若未提供路径，生成一张仿真加工表面。"""
    if path:
        print(f"\n载入表面图: {path}")
        try:
            import tifffile
            Z = tifffile.imread(path)
        except Exception:
            from PIL import Image
            Z = np.array(Image.open(path))
        # 压成二维高度图
        while Z.ndim > 2:
            Z = Z[:, :, 0] if Z.ndim == 3 else Z[0]
        Z = Z.astype(np.float64)
        # 去除明显的量纲（如 0-255），归一化到纳米级仿真量纲
        if Z.max() - Z.min() > 1000:
            Z = (Z - Z.mean()) * 1e-8 + 0.0
        px = 1e-6  # 假设横向分辨率 1 µm（请按你仪器的实际值修改）
        return Z, px

    # —— 仿真表面：低频波纹(走刀) + 高频随机粗糙度 ——
    print("\n未提供图片，自动生成仿真加工表面（256x256，1 µm 分辨率）")
    H = W = 256
    px = 1e-6  # 横向采样间距 1 µm
    x = np.linspace(0, 50e-6, W)
    y = np.linspace(0, 50e-6, H)
    X, Y = np.meshgrid(x, y)
    # 走刀波纹（低频）
    waviness = 2e-7 * np.sin(2 * np.pi * X / 20e-6) * np.cos(2 * np.pi * Y / 15e-6)
    # 随机粗糙度（高频），用高斯滤波模拟真实表面相关长度
    from scipy.ndimage import gaussian_filter
    rough = np.random.normal(0, 5e-8, (H, W))
    rough = gaussian_filter(rough, sigma=1.5)
    Z = waviness + rough
    return Z, px


# ---------- 2. 计算 ISO 25178 面粗糙度参数 ----------



# ---------- 3. 主流程 ----------
def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None
    Z, px = load_or_synth(path)

    H, W = Z.shape
    print(f"  尺寸: {W} x {H}   横向分辨率 px = {px*1e6:.3f} µm")

    Sa, Sq, Sz = areal_params(Z)
    print("\n" + "=" * 60)
    print("1. ISO 25178 面粗糙度参数")
    print("=" * 60)
    print(f"  Sa = {Sa*1e9:.3f} nm   (算术平均高度)")
    print(f"  Sq = {Sq*1e9:.3f} nm   (均方根高度)")
    print(f"  Sz = {Sz*1e9:.3f} nm   (十点高度 S10z)")

    # ---------- 4. 3D 表面图 ----------
    print("\n绘制 3D 表面图...")
    import matplotlib
    matplotlib.use("Agg")  # 无界面也能存图
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    xx, yy = np.meshgrid(np.arange(W) * px * 1e6, np.arange(H) * px * 1e6)  # µm
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(xx, yy, Z * 1e9, cmap="viridis", rstride=2, cstride=2,
                   linewidth=0, antialiased=True)
    ax.set_xlabel("X (µm)")
    ax.set_ylabel("Y (µm)")
    ax.set_zlabel("Z (nm)")
    ax.set_title(f"Surface topography  (Sa={Sa*1e9:.1f}nm, Sq={Sq*1e9:.1f}nm)")
    plt.tight_layout()
    fig.savefig("surface_3d.png", dpi=120)
    print("  已保存: surface_3d.png")

    # ---------- 5. 导出 Excel 报表（可选）----------
    try:
        from openpyxl import Workbook
        wb = Workbook()
        ws = wb.active
        ws.title = "Roughness"
        ws.append(["Parameter", "Value (nm)"])
        ws.append(["Sa", round(Sa * 1e9, 4)])
        ws.append(["Sq", round(Sq * 1e9, 4)])
        ws.append(["Sz", round(Sz * 1e9, 4)])
        ws.append(["Width_px", W])
        ws.append(["Height_px", H])
        ws.append(["Pixel_size_um", round(px * 1e6, 4)])
        wb.save("roughness_report.xlsx")
        print("  已保存: roughness_report.xlsx")
    except Exception as e:
        print(f"  (跳过 Excel 导出: {e})")

    print("\n✅ 完成。下一步可把真实表面图作为第一个参数传入，"
          "或在此脚本基础上加一个 CNN/UNet 做缺陷分割或粗糙度回归。")


if __name__ == "__main__":
    main()
