#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将Roboflow导出的VOC格式数据集转换为标准VOC格式
标准VOC格式目录结构：
    dataset_root/
    ├── Annotations/          # 所有XML标注文件
    ├── JPEGImages/          # 所有图片文件
    ├── ImageSets/
    │   └── Main/
    │       ├── train.txt    # 训练集图片ID列表（不带扩展名）
    │       ├── valid.txt    # 验证集图片ID列表（不带扩展名）
    │       └── test.txt     # 测试集图片ID列表（不带扩展名）
    └── labels.txt           # 类别列表文件
"""

import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import OrderedDict

try:
    from tqdm import tqdm
except ImportError:
    # 如果没有tqdm，使用简单的pass函数
    def tqdm(iterable, desc=""):
        print(desc)
        return iterable


def extract_classes_from_xml(xml_file):
    """从XML文件中提取类别名称"""
    tree = ET.parse(xml_file)
    root = tree.getroot()
    classes = []
    for obj in root.findall('object'):
        class_name = obj.find('name').text
        if class_name not in classes:
            classes.append(class_name)
    return classes


def update_xml_paths(xml_file, new_filename):
    """更新XML文件中的路径信息"""
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    # 更新filename
    filename_elem = root.find('filename')
    if filename_elem is not None:
        filename_elem.text = new_filename
    
    # 更新path
    path_elem = root.find('path')
    if path_elem is not None:
        path_elem.text = new_filename
    
    return tree


def convert_to_voc_format(source_dir, output_dir):
    """
    将Roboflow VOC格式转换为标准VOC格式
    
    Args:
        source_dir: 源数据集目录（包含train/, test/, valid/子目录）
        output_dir: 输出目录（将创建标准VOC结构）
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    # 创建输出目录结构
    annotations_dir = output_path / "Annotations"
    jpegimages_dir = output_path / "JPEGImages"
    imagesets_dir = output_path / "ImageSets" / "Main"
    
    annotations_dir.mkdir(parents=True, exist_ok=True)
    jpegimages_dir.mkdir(parents=True, exist_ok=True)
    imagesets_dir.mkdir(parents=True, exist_ok=True)
    
    # 收集所有类别
    all_classes = OrderedDict()
    
    # 处理train, test, valid目录
    splits = {}
    if (source_path / "train").exists():
        splits["train"] = list((source_path / "train").glob("*.xml"))
    if (source_path / "test").exists():
        splits["test"] = list((source_path / "test").glob("*.xml"))
    if (source_path / "valid").exists():
        splits["valid"] = list((source_path / "valid").glob("*.xml"))
    
    print(f"找到的数据集分割: {list(splits.keys())}")
    
    # 收集所有图片ID和类别
    all_image_ids = {"train": [], "test": [], "valid": []}
    
    # 处理train目录
    if "train" in splits:
        print("处理训练集...")
        for xml_file in tqdm(splits["train"]):
            # 提取类别
            classes = extract_classes_from_xml(xml_file)
            for cls in classes:
                all_classes[cls] = True
            
            # 获取对应的图片文件
            img_extensions = ['.jpg', '.jpeg', '.png']
            img_file = None
            for ext in img_extensions:
                candidate = xml_file.with_suffix(ext)
                if candidate.exists():
                    img_file = candidate
                    break
            
            if img_file is None:
                print(f"警告: 未找到 {xml_file} 对应的图片文件")
                continue
            
            # 生成新的文件名（使用不带扩展名的原始文件名）
            image_id = xml_file.stem
            new_xml_name = f"{image_id}.xml"
            new_img_name = f"{image_id}{img_file.suffix}"
            
            # 复制并更新XML文件
            new_xml_path = annotations_dir / new_xml_name
            tree = update_xml_paths(xml_file, new_img_name)
            tree.write(new_xml_path, encoding='utf-8', xml_declaration=True)
            
            # 复制图片文件
            new_img_path = jpegimages_dir / new_img_name
            shutil.copy2(img_file, new_img_path)
            
            all_image_ids["train"].append(image_id)
    
    # 处理test目录
    if "test" in splits:
        print("处理测试集...")
        for xml_file in tqdm(splits["test"]):
            # 提取类别
            classes = extract_classes_from_xml(xml_file)
            for cls in classes:
                all_classes[cls] = True
            
            # 获取对应的图片文件
            img_extensions = ['.jpg', '.jpeg', '.png']
            img_file = None
            for ext in img_extensions:
                candidate = xml_file.with_suffix(ext)
                if candidate.exists():
                    img_file = candidate
                    break
            
            if img_file is None:
                print(f"警告: 未找到 {xml_file} 对应的图片文件")
                continue
            
            # 生成新的文件名
            image_id = xml_file.stem
            new_xml_name = f"{image_id}.xml"
            new_img_name = f"{image_id}{img_file.suffix}"
            
            # 复制并更新XML文件
            new_xml_path = annotations_dir / new_xml_name
            tree = update_xml_paths(xml_file, new_img_name)
            tree.write(new_xml_path, encoding='utf-8', xml_declaration=True)
            
            # 复制图片文件
            new_img_path = jpegimages_dir / new_img_name
            shutil.copy2(img_file, new_img_path)
            
            all_image_ids["test"].append(image_id)
    
    # 处理valid目录（保持独立，不合并到train）
    if "valid" in splits:
        print("处理验证集...")
        for xml_file in tqdm(splits["valid"]):
            # 提取类别
            classes = extract_classes_from_xml(xml_file)
            for cls in classes:
                all_classes[cls] = True
            
            # 获取对应的图片文件
            img_extensions = ['.jpg', '.jpeg', '.png']
            img_file = None
            for ext in img_extensions:
                candidate = xml_file.with_suffix(ext)
                if candidate.exists():
                    img_file = candidate
                    break
            
            if img_file is None:
                print(f"警告: 未找到 {xml_file} 对应的图片文件")
                continue
            
            # 生成新的文件名
            image_id = xml_file.stem
            new_xml_name = f"{image_id}.xml"
            new_img_name = f"{image_id}{img_file.suffix}"
            
            # 复制并更新XML文件
            new_xml_path = annotations_dir / new_xml_name
            tree = update_xml_paths(xml_file, new_img_name)
            tree.write(new_xml_path, encoding='utf-8', xml_declaration=True)
            
            # 复制图片文件
            new_img_path = jpegimages_dir / new_img_name
            shutil.copy2(img_file, new_img_path)
            
            all_image_ids["valid"].append(image_id)
    
    # 写入ImageSets文件
    train_file = imagesets_dir / "train.txt"
    valid_file = imagesets_dir / "valid.txt"
    test_file = imagesets_dir / "test.txt"
    
    # train.txt 只包含 train
    train_ids = sorted(all_image_ids["train"])
    if train_ids:
        with open(train_file, 'w', encoding='utf-8') as f:
            for image_id in train_ids:
                f.write(f"{image_id}\n")
    
    # valid.txt 只包含 valid
    valid_ids = sorted(all_image_ids["valid"])
    if valid_ids:
        with open(valid_file, 'w', encoding='utf-8') as f:
            for image_id in valid_ids:
                f.write(f"{image_id}\n")
    
    # test.txt 只包含 test
    test_ids = sorted(all_image_ids["test"])
    if test_ids:
        with open(test_file, 'w', encoding='utf-8') as f:
            for image_id in test_ids:
                f.write(f"{image_id}\n")
    
    # 打印统计信息
    print(f"\n数据集分割统计:")
    print(f"  train目录: {len(splits.get('train', []))} 个文件")
    if "valid" in splits:
        print(f"  valid目录: {len(splits['valid'])} 个文件")
    print(f"  test目录: {len(splits.get('test', []))} 个文件")
    print(f"  -> train.txt: {len(train_ids)} 个图片ID")
    if valid_ids:
        print(f"  -> valid.txt: {len(valid_ids)} 个图片ID")
    print(f"  -> test.txt: {len(test_ids)} 个图片ID")
    
    # 写入labels.txt文件
    labels_file = output_path / "labels.txt"
    class_list = list(all_classes.keys())
    with open(labels_file, 'w', encoding='utf-8') as f:
        f.write(','.join(class_list))
    
    print(f"\n转换完成！")
    print(f"输出目录: {output_path}")
    print(f"类别列表: {class_list}")
    print(f"\n生成的文件:")
    print(f"  - 类别文件: {labels_file}")
    if train_ids:
        print(f"  - 训练集列表 (train.txt): {train_file}")
    if valid_ids:
        print(f"  - 验证集列表 (valid.txt): {valid_file}")
    if test_ids:
        print(f"  - 测试集列表 (test.txt): {test_file}")


if __name__ == "__main__":

    source = "Human_face_emotions"
    output = "voc_formatted_dataset"
    
    convert_to_voc_format(source, output)

