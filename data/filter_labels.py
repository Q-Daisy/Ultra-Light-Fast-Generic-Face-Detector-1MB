#!/usr/bin/env python3
"""
脚本用于从VOC格式数据集中删除不需要的标签类别
只保留：neutral, happy, anger, surprise, fear, sad
删除：disgust, content
"""
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict

# 配置路径
DATASET_ROOT = "/root/Project/Ultra-Light-Fast-Generic-Face-Detector-1MB/data/voc_formatted_dataset"
ANNOTATIONS_DIR = os.path.join(DATASET_ROOT, "Annotations")
IMAGESETS_DIR = os.path.join(DATASET_ROOT, "ImageSets/Main")
LABELS_FILE = os.path.join(DATASET_ROOT, "labels.txt")

# 要保留的类别（注意：用户写的是suprise，但实际应该是surprise）
KEEP_CLASSES = ["neutral", "happy", "anger", "surprise", "fear", "sad"]
# 要删除的类别
REMOVE_CLASSES = ["disgust", "content"]


def parse_xml(xml_path):
    """解析XML文件，返回根元素"""
    tree = ET.parse(xml_path)
    return tree.getroot()


def update_xml_file(xml_path, keep_classes):
    """
    更新XML文件，删除不在keep_classes列表中的object
    返回：(是否修改了文件, 保留的对象数量, 删除的对象数量)
    """
    root = parse_xml(xml_path)
    objects = root.findall('object')
    
    removed_count = 0
    kept_count = 0
    objects_to_remove = []
    
    # 找出需要删除的object
    for obj in objects:
        name_elem = obj.find('name')
        if name_elem is not None:
            class_name = name_elem.text.strip()
            if class_name not in keep_classes:
                objects_to_remove.append(obj)
                removed_count += 1
            else:
                kept_count += 1
    
    # 删除不需要的object
    if removed_count > 0:
        for obj in objects_to_remove:
            root.remove(obj)
        
        # 保存更新后的XML
        tree = ET.ElementTree(root)
        tree.write(xml_path, encoding='utf-8', xml_declaration=True)
        return True, kept_count, removed_count
    
    return False, kept_count, removed_count


def get_image_name_from_xml(xml_path):
    """从XML文件中提取图像名称（不含扩展名）"""
    root = parse_xml(xml_path)
    filename_elem = root.find('filename')
    if filename_elem is not None:
        filename = filename_elem.text
        # 去掉扩展名
        return os.path.splitext(filename)[0]
    return None


def update_labels_file(labels_file, keep_classes):
    """更新labels.txt文件"""
    new_labels = ','.join(keep_classes)
    with open(labels_file, 'w') as f:
        f.write(new_labels)
    print(f"✓ 已更新 {labels_file}")
    print(f"  新标签: {new_labels}")


def update_imagesets(imagesets_dir, removed_xml_files):
    """更新ImageSets中的txt文件，移除已删除的文件"""
    removed_names = set()
    for xml_file in removed_xml_files:
        name = get_image_name_from_xml(xml_file)
        if name:
            removed_names.add(name)
    
    if not removed_names:
        return
    
    txt_files = ['train.txt', 'valid.txt', 'test.txt']
    for txt_file in txt_files:
        txt_path = os.path.join(imagesets_dir, txt_file)
        if not os.path.exists(txt_path):
            continue
        
        with open(txt_path, 'r') as f:
            lines = f.readlines()
        
        # 过滤掉被删除的文件
        filtered_lines = [line.strip() for line in lines if line.strip() not in removed_names]
        
        # 写回文件
        with open(txt_path, 'w') as f:
            for line in filtered_lines:
                f.write(line + '\n')
        
        removed_count = len(lines) - len(filtered_lines)
        if removed_count > 0:
            print(f"✓ 已从 {txt_file} 中移除 {removed_count} 个文件")


def main():
    print("=" * 60)
    print("数据集标签过滤脚本")
    print("=" * 60)
    print(f"保留的类别: {', '.join(KEEP_CLASSES)}")
    print(f"删除的类别: {', '.join(REMOVE_CLASSES)}")
    print(f"数据集路径: {DATASET_ROOT}")
    print("=" * 60)
    
    # 检查路径
    if not os.path.exists(ANNOTATIONS_DIR):
        print(f"错误: 找不到Annotations目录: {ANNOTATIONS_DIR}")
        return
    
    # 获取所有XML文件
    xml_files = list(Path(ANNOTATIONS_DIR).glob("*.xml"))
    total_files = len(xml_files)
    print(f"\n找到 {total_files} 个XML文件")
    
    # 统计信息
    stats = {
        'modified': 0,
        'removed_objects': 0,
        'kept_objects': 0,
        'empty_files': 0,
        'unchanged': 0
    }
    
    removed_xml_files = []  # 记录完全空的文件（可选：是否删除）
    
    print("\n开始处理XML文件...")
    for i, xml_path in enumerate(xml_files, 1):
        if i % 500 == 0:
            print(f"  处理进度: {i}/{total_files} ({i*100//total_files}%)")
        
        modified, kept_count, removed_count = update_xml_file(str(xml_path), KEEP_CLASSES)
        
        if modified:
            stats['modified'] += 1
            stats['kept_objects'] += kept_count
            stats['removed_objects'] += removed_count
            
            # 如果删除后没有对象了，记录该文件
            if kept_count == 0:
                stats['empty_files'] += 1
                removed_xml_files.append(str(xml_path))
        else:
            stats['unchanged'] += 1
            stats['kept_objects'] += kept_count
    
    print(f"\n处理完成！")
    print("=" * 60)
    print("统计信息:")
    print(f"  总文件数: {total_files}")
    print(f"  修改的文件: {stats['modified']}")
    print(f"  未修改的文件: {stats['unchanged']}")
    print(f"  保留的对象数: {stats['kept_objects']}")
    print(f"  删除的对象数: {stats['removed_objects']}")
    print(f"  空文件数（删除后无对象）: {stats['empty_files']}")
    print("=" * 60)
    
    # 更新labels.txt
    print("\n更新labels.txt文件...")
    update_labels_file(LABELS_FILE, KEEP_CLASSES)
    
    # 更新ImageSets文件
    if removed_xml_files:
        print(f"\n更新ImageSets文件（移除 {len(removed_xml_files)} 个空文件）...")
        update_imagesets(IMAGESETS_DIR, removed_xml_files)
    
    print("\n✓ 所有处理完成！")
    print("\n提示:")
    print("  - 空文件（删除后无对象的XML）仍然保留在Annotations目录中")
    print("  - 如果需要删除空文件，可以手动删除或使用其他脚本")
    print("  - 训练时，空的XML文件通常会被忽略")


if __name__ == "__main__":
    main()

