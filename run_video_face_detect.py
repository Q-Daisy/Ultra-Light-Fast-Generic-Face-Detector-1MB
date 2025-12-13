"""
在WSL中使用网络摄像头流的版本
需要在Windows中先运行 camera_server_windows.py
"""
import argparse
import sys
import cv2
import requests
import numpy as np
import subprocess
import socket
from io import BytesIO
from PIL import Image

from vision.ssd.config.fd_config import define_img_size

parser = argparse.ArgumentParser(
    description='detect_video from network camera stream')

parser.add_argument('--net_type', default="RFB", type=str,
                    help='The network architecture ,optional: RFB (higher precision) or slim (faster)')
parser.add_argument('--input_size', default=320, type=int,
                    help='define network input size,default optional value 128/160/320/480/640/1280')
parser.add_argument('--threshold', default=0.7, type=float,
                    help='score threshold')
parser.add_argument('--candidate_size', default=1000, type=int,
                    help='nms candidate size')
parser.add_argument('--test_device', default="cuda:0", type=str,
                    help='cuda:0 or cpu')
parser.add_argument('--camera_url', default="", type=str,
                    help='URL of camera stream (leave empty to auto-detect Windows IP)')
parser.add_argument('--camera_port', default=8081, type=int,
                    help='Port of camera server (default: 8080)')
args = parser.parse_args()

input_img_size = args.input_size
define_img_size(input_img_size)

from vision.ssd.mb_tiny_fd import create_mb_tiny_fd, create_mb_tiny_fd_predictor
from vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd, create_Mb_Tiny_RFB_fd_predictor
from vision.utils.misc import Timer

label_path = "./models/RFB-finetuned-emotion-320/emotion-labels.txt"
net_type = args.net_type

class_names = [name.strip() for name in open(label_path).readlines()]
num_classes = len(class_names)
test_device = args.test_device

candidate_size = args.candidate_size
threshold = args.threshold

if net_type == 'slim':
    model_path = "models/pretrained/version-slim-320.pth"
    net = create_mb_tiny_fd(len(class_names), is_test=True, device=test_device)
    predictor = create_mb_tiny_fd_predictor(net, candidate_size=candidate_size, device=test_device)
elif net_type == 'RFB':
    # model_path = "models/pretrained/version-RFB-320.pth"
    # model_path = "models/pretrained/RFB-age1-best.pth"
    model_path = "models/RFB-finetuned-emotion-320/RFB-Emotion2.pth"
    net = create_Mb_Tiny_RFB_fd(len(class_names), is_test=True, device=test_device)
    predictor = create_Mb_Tiny_RFB_fd_predictor(net, candidate_size=candidate_size, device=test_device)
else:
    print("The net type is wrong!")
    sys.exit(1)

# 尝试加载模型，如果类别数量不匹配，使用非严格模式
try:
    net.load(model_path, strict=True)
except RuntimeError as e:
    if "size mismatch" in str(e):
        print(f"警告：模型类别数量不匹配，尝试部分加载...")
        print(f"当前模型类别数: {len(class_names)}, 检查点类别数可能不同")
        net.load(model_path, strict=False)
        print("已加载兼容的权重，分类头将使用随机初始化")
    else:
        raise

# 自动检测Windows IP地址
def get_windows_ip():
    """从WSL中获取Windows主机的IP地址"""
    import subprocess
    try:
        # 方法1: 从/etc/resolv.conf获取
        with open('/etc/resolv.conf', 'r') as f:
            for line in f:
                if line.startswith('nameserver'):
                    ip = line.split()[1]
                    # 验证是否是有效的IP
                    parts = ip.split('.')
                    if len(parts) == 4 and all(0 <= int(p) <= 255 for p in parts):
                        return ip
    except:
        pass
    
    try:
        # 方法2: 使用hostname命令
        result = subprocess.run(['hostname', '-I'], capture_output=True, text=True)
        if result.returncode == 0:
            ips = result.stdout.strip().split()
            # 通常Windows IP是第一个，但需要验证
            for ip in ips:
                parts = ip.split('.')
                if len(parts) == 4 and all(0 <= int(p) <= 255 for p in parts):
                    # 尝试连接，如果能连接就是Windows IP
                    return ip
    except:
        pass
    
    return None

# 确定摄像头URL
if args.camera_url:
    camera_url = args.camera_url
else:
    # 自动检测Windows IP
    windows_ip = get_windows_ip()
    if windows_ip:
        camera_url = f"http://{windows_ip}:{args.camera_port}/video"
        print(f"自动检测到Windows IP: {windows_ip}")
    else:
        # 尝试常见的WSL到Windows的IP
        # WSL2中，Windows主机通常在.2地址
        import socket
        hostname = socket.gethostname()
        try:
            # 尝试获取网关IP（通常是Windows IP）
            result = subprocess.run(['ip', 'route', 'show'], capture_output=True, text=True)
            for line in result.stdout.split('\n'):
                if 'default' in line:
                    parts = line.split()
                    if len(parts) > 2:
                        windows_ip = parts[2]
                        camera_url = f"http://{windows_ip}:{args.camera_port}/video"
                        print(f"从路由表获取Windows IP: {windows_ip}")
                        break
            else:
                # 如果都失败了，使用默认值
                camera_url = f"http://localhost:{args.camera_port}/video"
                print(f"警告：无法自动检测Windows IP，使用localhost（可能无法连接）")
        except:
            camera_url = f"http://localhost:{args.camera_port}/video"
            print(f"警告：无法自动检测Windows IP，使用localhost（可能无法连接）")

print(f"连接到摄像头流: {camera_url}")
print("如果连接失败，请手动指定Windows IP地址：")
print("  python run_video_face_detect_wsl.py --camera_url http://<Windows_IP>:<port>/video")
print("  例如: python run_video_face_detect_wsl.py --camera_url http://10.32.207.28:8081/video")
print("按 'q' 退出")

timer = Timer()
sum_faces = 0

try:
    stream = requests.get(camera_url, stream=True, timeout=5)
    if stream.status_code != 200:
        print(f"无法连接到摄像头流: {camera_url}")
        print("请确保在Windows中运行了 camera_server_windows.py")
        print(f"Windows服务器应该运行在端口 {args.camera_port}")
        sys.exit(1)
except requests.exceptions.ConnectionError as e:
    print(f"连接错误: 无法连接到 {camera_url}")
    print("\n解决方案：")
    print("1. 确保Windows中运行了 camera_server_windows.py")
    print("2. 检查Windows防火墙是否允许该端口")
    print("3. 手动指定Windows IP地址：")
    print(f"   python run_video_face_detect_wsl.py --camera_url http://<Windows_IP>:{args.camera_port}/video")
    print("\n获取Windows IP的方法（在Windows PowerShell中）：")
    print("   ipconfig | findstr IPv4")
    sys.exit(1)
except Exception as e:
    print(f"连接错误: {e}")
    print("请确保在Windows中运行了 camera_server_windows.py")
    sys.exit(1)

bytes_data = b''
while True:
    try:
        chunk = stream.raw.read(1024)
        if not chunk:
            break
        bytes_data += chunk
        a = bytes_data.find(b'\xff\xd8')  # JPEG开始标记
        b = bytes_data.find(b'\xff\xd9')  # JPEG结束标记
        
        if a != -1 and b != -1:
            jpg = bytes_data[a:b+2]
            bytes_data = bytes_data[b+2:]
            
            # 解码图像
            img = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            
            image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            timer.start()
            boxes, labels, probs = predictor.predict(image, candidate_size / 2, threshold)
            interval = timer.end()
            
            # 打印检测结果（包含类别信息）
            detected_classes = []
            for i in range(boxes.size(0)):
                class_idx = labels[i].item()
                class_name = class_names[class_idx] if class_idx < len(class_names) else f"Class_{class_idx}"
                prob = probs[i].item()
                detected_classes.append(f"{class_name}({prob:.2f})")
            
            print('Time: {:.6f}s, Detect Objects: {:d}. Classes: {}'.format(
                interval, labels.size(0), ', '.join(detected_classes) if detected_classes else 'None'))
            
            # 在图像上绘制边界框和类别标签
            for i in range(boxes.size(0)):
                box = boxes[i, :]
                class_idx = labels[i].item()
                class_name = class_names[class_idx] if class_idx < len(class_names) else f"Class_{class_idx}"
                prob = probs[i].item()
                
                # 绘制边界框
                cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 4)
                
                # 准备标签文本
                label_text = f"{class_name}: {prob:.2f}"
                
                # 计算文本大小和位置
                (text_width, text_height), baseline = cv2.getTextSize(
                    label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                
                # 计算文本位置（优先放在边界框上方）
                label_y = int(box[1]) - text_height - baseline - 5
                if label_y < 0:
                    # 如果上方空间不够，放在下方
                    label_y = int(box[3]) + text_height + baseline + 5
                
                # 绘制文本背景（绿色矩形）
                cv2.rectangle(img, 
                             (int(box[0]), label_y - text_height - baseline - 2),
                             (int(box[0]) + text_width, label_y + baseline),
                             (0, 255, 0), -1)
                
                # 绘制文本（黑色，粗体）
                cv2.putText(img, label_text,
                           (int(box[0]), label_y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            sum_faces += boxes.size(0)
            img = cv2.resize(img, None, None, fx=0.8, fy=0.8)
            cv2.imshow('annotated', img)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except KeyboardInterrupt:
        break
    except Exception as e:
        print(f"处理错误: {e}")
        break

print(f"all face num: {sum_faces}")

