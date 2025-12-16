"""
This code performs real-time detection using a webcam and displays FPS.
Optimized for better accuracy and stability.
"""
import argparse
import sys
import time
import cv2
import torch
import random
import numpy as np
import os
from collections import deque

from vision.ssd.config.fd_config import define_img_size

# 自动检测并设置 DISPLAY 环境变量
def setup_display():
    """尝试自动设置 DISPLAY 环境变量并检测是否可以显示窗口"""
    if not os.environ.get('DISPLAY'):
        # 检查常见的显示位置
        if os.path.exists('/tmp/.X11-unix/X0'):
            os.environ['DISPLAY'] = ':0'
            print(f"自动设置 DISPLAY=:0")
        elif os.path.exists('/tmp/.X11-unix/X1'):
            os.environ['DISPLAY'] = ':1'
            print(f"自动设置 DISPLAY=:1")
    
    # 检测是否可以显示窗口
    try:
        test_img = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.namedWindow('test', cv2.WINDOW_NORMAL)
        cv2.imshow('test', test_img)
        cv2.waitKey(1)
        cv2.destroyWindow('test')
        return True
    except Exception as e:
        # 静默失败，返回 False
        return False

# 在导入后立即检测
HAS_DISPLAY = setup_display()

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True


def calculate_iou(box1, box2):
    """计算两个框的IoU"""
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # 计算交集区域
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
        return 0.0
    
    inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0.0
    return inter_area / union_area


def nms_filter(boxes, probs, labels, iou_threshold=0.4):
    """额外的NMS过滤，去除重复框"""
    if len(boxes) == 0:
        return boxes, probs, labels
    
    # 按置信度排序
    indices = np.argsort(probs)[::-1]
    keep = []
    
    while len(indices) > 0:
        current_idx = indices[0]
        keep.append(current_idx)
        
        if len(indices) == 1:
            break
        
        current_box = boxes[current_idx]
        
        # 计算与剩余框的IoU
        ious = []
        for idx in indices[1:]:
            iou = calculate_iou(
                [current_box[0], current_box[1], current_box[2], current_box[3]],
                [boxes[idx][0], boxes[idx][1], boxes[idx][2], boxes[idx][3]]
            )
            ious.append(iou)
        
        # 保留IoU小于阈值的框（即不重叠的框）
        mask = np.array(ious) < iou_threshold
        indices = indices[1:][mask]
    
    keep = np.array(keep)
    return boxes[keep], probs[keep], labels[keep]


def filter_boxes_by_size(boxes, probs, labels, img_width, img_height, min_area=400, max_area_ratio=0.5, min_aspect_ratio=0.3, max_aspect_ratio=3.0):
    """根据面积和宽高比过滤框"""
    if len(boxes) == 0:
        return boxes, probs, labels
    
    img_area = img_width * img_height
    valid_indices = []
    
    for i, box in enumerate(boxes):
        x_min, y_min, x_max, y_max = float(box[0]), float(box[1]), float(box[2]), float(box[3])
        width = x_max - x_min
        height = y_max - y_min
        area = width * height
        
        # 检查最小面积
        if area < min_area:
            continue
        
        # 检查最大面积（避免检测到整张图像）
        if area > img_area * max_area_ratio:
            continue
        
        # 检查宽高比（人脸通常是接近正方形的）
        if width == 0 or height == 0:
            continue
        aspect_ratio = width / height
        if aspect_ratio < min_aspect_ratio or aspect_ratio > max_aspect_ratio:
            continue
        
        valid_indices.append(i)
    
    if len(valid_indices) == 0:
        return np.array([]), np.array([]), np.array([])
    
    valid_indices = np.array(valid_indices)
    return boxes[valid_indices], probs[valid_indices], labels[valid_indices]


def detect(args, predictor, class_names):
    if args.on_board:
        from jetcam.csi_camera import CSICamera
        from jetcam.usb_camera import USBCamera
        #cap = CSICamera(capture_device=0, width=args.width, height=args.height)
        cap = USBCamera(capture_device=0, width=args.width, height=args.height)
        # 启用后台线程模式，持续捕获最新帧（关键优化：减少延迟）
        cap.running = True
        print("摄像头初始化完成，等待启动...")
        time.sleep(0.5)  # 等待摄像头后台线程启动
    else:
        cap = cv2.VideoCapture(0) 
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
        # 对于 USB 摄像头，设置缓冲区大小为1以减少延迟
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap:
        print("Error: Unable to open video file or camera stream.")
        sys.exit(1)
    
    if args.display and HAS_DISPLAY:
        print("Starting real-time detection. Press 'q' to quit.")
    else:
        if not args.display:
            print("Starting real-time detection (无显示模式). 按 Ctrl+C 退出.")
        else:
            print("警告: 无法连接到显示服务器，切换到无显示模式. 按 Ctrl+C 退出.")
            args.display = False
    his_fps = []
    try:
        while True:
            t = time.time() 
            if args.on_board:
                # 直接读取最新帧（后台线程已更新，无需等待）
                frame = cap.value
                if frame is None:
                    continue  # 如果还没有帧，跳过本次循环
                frame = frame.copy()  # 复制数据以避免线程安全问题
            else:
                ret, frame = cap.read()
                if not ret or frame is None:
                    print("Error: Unable to read frame from camera stream.")
                    break
            
            image = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB)

            # 检测
            boxes, labels, probs = predictor.predict(image, args.candidate_size // 2, args.threshold)
            
            # 后处理优化：过滤和去重
            if boxes.size(0) > 0 and probs.size(0) > 0:
                # 转换为numpy以便处理
                boxes_np = boxes.cpu().numpy()
                probs_np = probs.cpu().numpy()
                labels_np = labels.cpu().numpy() if isinstance(labels, torch.Tensor) else np.array(labels)
                
                # 1. 面积和宽高比过滤
                h, w = frame.shape[:2]
                boxes_np, probs_np, labels_np = filter_boxes_by_size(
                    boxes_np, probs_np, labels_np, w, h,
                    min_area=args.min_area,
                    max_area_ratio=args.max_area_ratio,
                    min_aspect_ratio=args.min_aspect_ratio,
                    max_aspect_ratio=args.max_aspect_ratio
                )
                
                # 2. 二次NMS过滤（更严格的IoU阈值）
                if len(boxes_np) > 0:
                    boxes_np, probs_np, labels_np = nms_filter(
                        boxes_np, probs_np, labels_np,
                        iou_threshold=args.nms_iou_threshold
                    )
                    
                    # 3. 再次按置信度排序，只保留top-k
                    if len(boxes_np) > 0:
                        top_k = args.max_detections
                        if len(probs_np) > top_k:
                            top_indices = np.argsort(probs_np)[::-1][:top_k]
                            boxes_np = boxes_np[top_indices]
                            probs_np = probs_np[top_indices]
                            labels_np = labels_np[top_indices]
                        
                        # 转换回torch tensor
                        device = boxes.device
                        boxes = torch.from_numpy(boxes_np).float().to(device)
                        probs = torch.from_numpy(probs_np).float().to(device)
                        labels = torch.from_numpy(labels_np).long().to(device)
                else:
                    boxes = torch.tensor([])
                    probs = torch.tensor([])
                    labels = torch.tensor([])
            else:
                boxes = torch.tensor([])
                probs = torch.tensor([])
                labels = torch.tensor([])
            
            # 绘制结果
            if boxes.size(0) > 0 and labels.size(0) > 0:
                for i in range(boxes.size(0)):
                    box = boxes[i, :]
                    cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
                    label_idx = labels[i].item() if isinstance(labels[i], torch.Tensor) else int(labels[i])
                    label = f"{class_names[label_idx]}: {probs[i]:.2f}"
                    cv2.putText(frame, label, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.putText(frame, f"Count: {boxes.size(0)}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            fps = 1 / (time.time() - t)
            his_fps.append(fps)
            #cv2.putText(frame, f"FPS: {fps:.2f}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            delay=(time.time() - t)
            cv2.putText(frame, f"delay: {delay:.2f}", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
            
            # 显示窗口（如果启用且可用）
            if args.display and HAS_DISPLAY:
                cv2.imshow("Real-time Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                # 无显示模式：打印统计信息
                if len(his_fps) % 30 == 0:  # 每30帧打印一次
                    print(f"检测到 {boxes.size(0)} 个目标, FPS: {fps:.2f}, Delay: {delay:.3f}s")
                # 检查是否应该退出（通过信号或其他方式）
                time.sleep(0.01)  # 避免CPU占用过高
            if len(his_fps) >= 1000000:
                print(f"Mean fps is {sum(his_fps) / len(his_fps)}")
                his_fps = []
    except KeyboardInterrupt:
        print("\n检测中断")
    finally:
        print(f"Mean fps is {sum(his_fps) / len(his_fps) if his_fps else 0:.2f}")
        
        # 正确清理资源
        if args.on_board:
            # 停止后台线程
            cap.running = False
        else:
            cap.release()
        if args.display and HAS_DISPLAY:
            cv2.destroyAllWindows()

def main(args):
    # device = "cpu"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    define_img_size(args.input_size)
    class_names = [name.strip() for name in open(args.label_path).readlines()]
    
    if args.net_type == 'slim':
        model_path = args.model_path
        net = create_mb_tiny_fd(len(class_names), is_test=True, device=device)
        predictor = create_mb_tiny_fd_predictor(net, candidate_size=args.candidate_size, device=device)
    elif args.net_type == 'RFB':
        model_path = args.model_path
        net = create_Mb_Tiny_RFB_fd(len(class_names), is_test=True, device=device)
        predictor = create_Mb_Tiny_RFB_fd_predictor(net, candidate_size=args.candidate_size, device=device)
    else:
        print("The net type is wrong!")
        sys.exit()
    net.load(model_path)
    net.float()
    
    detect(args, predictor, class_names)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Real-time object detection with FPS display')
    parser.add_argument('--net_type', default="RFB", type=str, help='The network architecture, optional: RFB or slim')
    parser.add_argument('--input_size', default=320, type=int, help='Network input size, e.g., 128/160/320/480/640/1280')
    parser.add_argument('--threshold', default=0.7, type=float, help='Score threshold')
    parser.add_argument('--candidate_size', default=1000, type=int, help='NMS candidate size')
    parser.add_argument('--on_board', default=True, action='store_true',help='Run on board')
    parser.add_argument('--width', default=640, help='Width of camera')
    parser.add_argument('--height', default=360, help='Height of camera')
    parser.add_argument('--model_path', default="./checkpoints/e95.pth", help='Path to the trained model')
    parser.add_argument('--label_path', default="./checkpoints/emotion-labels.txt", help='Path to the labels')
    
    # 新增的精度优化参数
    parser.add_argument('--nms_iou_threshold', default=0.4, type=float, help='二次NMS的IoU阈值（越大越严格，减少重复框）')
    parser.add_argument('--min_area', default=400, type=int, help='检测框最小面积（像素）')
    parser.add_argument('--max_area_ratio', default=0.5, type=float, help='检测框最大面积比例（相对于图像）')
    parser.add_argument('--min_aspect_ratio', default=0.3, type=float, help='最小宽高比')
    parser.add_argument('--max_aspect_ratio', default=3.0, type=float, help='最大宽高比')
    parser.add_argument('--max_detections', default=10, type=int, help='最大检测数量（top-k）')
    parser.add_argument('--display', default=None, action='store_true', help='启用显示窗口（默认：自动检测）')
    parser.add_argument('--no-display', dest='display', action='store_false', help='禁用显示窗口')
    args = parser.parse_args()
    
    # 如果没有明确指定，根据环境自动决定
    if args.display is None:
        args.display = HAS_DISPLAY
    define_img_size(args.input_size)
    from vision.ssd.mb_tiny_fd import create_mb_tiny_fd, create_mb_tiny_fd_predictor
    from vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd, create_Mb_Tiny_RFB_fd_predictor
    main(args)
