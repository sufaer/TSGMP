import os
import glob
import numpy as np
import imageio.v2 as imageio
from PIL import Image, ImageOps
import pandas as pd
import nibabel as nib
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("maxpng_dwi.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# 辅助函数：读取 NIfTI 文件
def read_niifile(niifilepath):
    """读取 NIfTI 文件并返回其数据"""
    try:
        return nib.load(niifilepath).get_fdata()
    except Exception as e:
        raise IOError(f"无法读取 NIfTI 文件 {niifilepath}: {e}")

# 辅助函数：查找 NIfTI 文件
def find_nii_file(seq_dir):
    """查找目录中的NIfTI文件，优先返回.nii文件"""
    nii_files = glob.glob(os.path.join(seq_dir, "*.nii"))
    nii_gz_files = glob.glob(os.path.join(seq_dir, "*.nii.gz"))
    all_nii_files = nii_files + nii_gz_files
    if not all_nii_files:
        raise FileNotFoundError(f"在目录 {seq_dir} 中未找到NIfTI文件")

    for file in all_nii_files:
        if file.endswith('.nii'):
            return file
    return all_nii_files[0]

# 核心函数：读取 NIfTI 数据并生成 Z 方向的内存切片
def get_z_slices_in_memory(niifilepath, channel=0):
    """
    读取 NIfTI 文件的所有 Z 方向切片，归一化并转换为 uint8，
    然后以列表形式返回这些切片。不保存到磁盘。
    
    Args:
        niifilepath: NIfTI 文件路径
        channel: 对于多通道数据，选择哪个通道（默认为0）
    """
    fdata = read_niifile(niifilepath)
    
    # 处理多维数据
    logger.info(f"检测到 {len(fdata.shape)} 维数据，形状: {fdata.shape}")
    
    # 对于5D数据 (x, y, z, t, c) - 取第一个时间点和指定通道
    if len(fdata.shape) == 5:
        logger.info(f"处理5D数据: 使用第一个时间点和通道 {channel}")
        fdata = fdata[:, :, :, 0, channel]  # 取第一个时间点和指定通道
    
    # 对于4D数据 (x, y, z, t) - 取第一个时间点
    elif len(fdata.shape) == 4:
        logger.info(f"处理4D数据: 使用第一个时间点")
        fdata = fdata[:, :, :, 0]  # 取第一个时间点
    
    # 确保现在是3D数据
    if len(fdata.shape) != 3:
        raise ValueError(f"处理后数据维度仍不是3D: {fdata.shape}")
    
    logger.info(f"处理后数据形状: {fdata.shape}")
    
    (x, y, z) = fdata.shape
    slices = []
    for k in range(z):
        slice_data = fdata[:, :, k]

        if np.max(slice_data) > 0:
            slice_data = (slice_data - np.min(slice_data)) / (np.max(slice_data) - np.min(slice_data)) * 255
        slice_data = slice_data.astype(np.uint8)
        slices.append(slice_data)
    return slices

# 函数：找到病灶区域最大的切片索引
def find_largest_lesion_slice_index(mask_slices):
    """
    遍历内存中的 mask 切片序列，找到病灶区域最大的切片索引。
    返回最大病灶切片的索引和最大面积。
    """
    max_lesion_area = -1
    largest_lesion_slice_index = None

    for i, mask in enumerate(mask_slices):
        if np.all(mask == 0):
            continue

        current_lesion_area = np.sum(mask > 0)

        if current_lesion_area > max_lesion_area:
            max_lesion_area = current_lesion_area
            largest_lesion_slice_index = i

    return largest_lesion_slice_index, max_lesion_area

# 函数：处理单个切片的 ROI (现在直接接收 numpy 数组)
def process_single_slice_roi_in_memory(original_slice, mask_slice, output_path, margin=50, output_size=(256, 256)):
    """
    处理单个 MRI 和 mask numpy 数组：
    1. 提取ROI（带margin）
    2. 输出指定大小的PNG
    """
    if np.all(mask_slice == 0):
        logger.info(f"警告: 给定的 mask 切片是全黑，不进行处理。")
        return

    mask_binary = (mask_slice > 0).astype(bool)
    rows = np.any(mask_binary, axis=1)
    cols = np.any(mask_binary, axis=0)

    if not np.any(rows) or not np.any(cols):
        logger.info(f"警告: 给定的 mask 切片无效，不进行处理。")
        return

    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]

    ymin = max(0, ymin - margin)
    ymax = min(original_slice.shape[0], ymax + margin)
    xmin = max(0, xmin - margin)
    xmax = min(original_slice.shape[1], xmax + margin)

    roi = original_slice[ymin:ymax, xmin:xmax]

    pil_roi = Image.fromarray(roi)
    pil_roi = ImageOps.pad(pil_roi, size=output_size, color=0)

    pil_roi.save(output_path)
    logger.info(f"已处理并保存 ROI 到: {output_path} | ROI尺寸: {roi.shape} → {output_size[0]}x{output_size[1]}")


def process_mri_data_no_intermediate_png(csv_file_path, base_nii_data_root, final_output_dir):
    """
    整合 MRI 数据处理流程，不保存中间 PNG 切片。

    Args:
        csv_file_path (str): 包含患者信息和文件路径的 CSV 文件路径。
        base_nii_data_root (str): 原始 NIfTI 数据根目录。
        final_output_dir (str): 最终处理后的 ROI 图像保存目录。
    """
    df = pd.read_csv(csv_file_path, encoding='gbk')

    os.makedirs(final_output_dir, exist_ok=True) # 确保最终输出目录存在

    for index, row in df.iterrows():
        patient_id = row['patient_ID']
        c1_mask_path_relative = row['c1']
        dwi1_mask_path_relative = row['dwi1']

        logger.info(f"--- 正在处理患者: {patient_id} ---")

        dce_nii_original_path = os.path.join(base_nii_data_root, f"{patient_id}_c1.nii.gz")
        dwi_nii_original_path = os.path.join(base_nii_data_root, f"{patient_id}_dwi1.nii.gz")

        dce_mask_nii_dir = c1_mask_path_relative
        dwi_mask_nii_dir = dwi1_mask_path_relative

        try:
            dce_mask_nii_path = find_nii_file(dce_mask_nii_dir)
            dwi_mask_nii_path = find_nii_file(dwi_mask_nii_dir)
        except FileNotFoundError as e:
            logger.info(f"错误: {e}。跳过患者 {patient_id}。")
            continue

        # 将 NIfTI 数据加载到内存中的切片列表
        logger.info(f"  正在加载 DCE 原始数据到内存...")
        dce_original_slices = get_z_slices_in_memory(dce_nii_original_path, channel=0)
        logger.info(f"  正在加载 DCE mask 数据到内存...")
        dce_mask_slices = get_z_slices_in_memory(dce_mask_nii_path, channel=0)

        logger.info(f"  正在加载 DWI 原始数据到内存...")
        # dwi_original_slices = get_z_slices_in_memory(dwi_nii_original_path, channel=0)
        logger.info(f"  正在加载 DWI mask 数据到内存...")
        # dwi_mask_slices = get_z_slices_in_memory(dwi_mask_nii_path, channel=0)

        modalities = {
            'c1': {'original_slices': dce_original_slices, 'mask_slices': dce_mask_slices},
            # 'dwi1': {'original_slices': dwi_original_slices, 'mask_slices': dwi_mask_slices}
        }

        for mod_name, slices_data in modalities.items():
            original_slices = slices_data['original_slices']
            mask_slices = slices_data['mask_slices']

            logger.info(f"  正在查找 {mod_name} 的最大病灶切片索引...")
            largest_slice_index, max_area = find_largest_lesion_slice_index(mask_slices)

            if largest_slice_index is not None:
                selected_original_slice = original_slices[largest_slice_index]
                selected_mask_slice = mask_slices[largest_slice_index]

                final_output_image_name = f"{patient_id}_{mod_name}.png"
                final_output_image_path = os.path.join(final_output_dir, final_output_image_name)

                logger.info(f"  正在处理 {mod_name} 最大病灶切片索引: {largest_slice_index} (面积: {max_area})...")

                # 这里的 margin 不同序列用不同的 margin ，DCE用 margin=10，DWI用 margin=5.
                process_single_slice_roi_in_memory(selected_original_slice, selected_mask_slice, final_output_image_path, margin=10)
            else:
                logger.info(f" 患者 {patient_id} 的 {mod_name} 没有找到有效病灶切片或所有 mask 都是全黑。")

        logger.info(f"--- 患者 {patient_id} 处理完成 ---\n")

# 示例用法
if __name__ == "__main__":
    csv_file = '/home/zhoutz/MY_Project/Again_2d/dataset_csv/new_1292.csv'
    original_nii_root = '/home/zhoutz/datasets/Dicomtonii'
    final_roi_output_dir = '/home/zhoutz/datasets/Croppng'
    
    # 运行处理函数
    process_mri_data_no_intermediate_png(csv_file, original_nii_root, final_roi_output_dir)

    logger.info("\n所有患者数据处理完成，未保存中间 PNG 切片！")