# Facial Recognition Model Featuring Emotional Detection

## 1. Introduction

This is a light-weighted deep learning project that focuses on facial recognition. Run this project to recognize faces and tell his/her emotion!

## 2. Installation and Usage Instructions

### 2.1 Installation Steps

1. **Clone the Repository**

   ```
   git clone https://github.com/Q-Daisy/Ultra-Light-Fast-Generic-Face-Detector-1MB.git
   ```

2. **Install Dependencies**  

   ```
   cd Ultra-Light-Fast-Generic-Face-Detector-1MB/
   pip install -r requirements.txt
   ```

   Please download torch manually according to your device requirements.
   For jetson installing,please follow NVIDIA's official guide:
   https://developer.nvidia.com/embedded/learn/get-started-jetson-nano-devkit

## 3. Examples and Code Snippets

### 3.1 Usage Examples

There are several parameters you can decide and run with. But with Quick-Start, you can run the script directly with the following command. Then you can test the Emotion model by default. 

```cmd
bash run.sh 
```

Also, you can change the parameters input and run other models or with other performance.

- `--model_path`: Path to the trained model
- `--label_path`: Path to the labels (Please ensure the label are corresponding to the model)  
- `--input_size`: Decide the size of the input image , e.g 320/640 (Also please ensure the size is corresponding to the pretrained model) 

## 4.Project Structure and File Organization

### 4.1 the File and Directory Structure

| File/Repository | Description                                         |
| :-------------- | :-------------------------------------------------- |
| /checkpoints    | Directory for storing pretrained models             |
| /data           | Directory for training data                         |
| /jetcam         | Directory for camera scripts                        |
| /models         | Directory for finetuned model parameters and labels |
| /vision         | Directory for model structural codes                |
| README.md       | Project description                                 |
| run.py          | For running the model                               |
| train.py        | For finetuning the model                            |

### 4.2 Purpose of Each File and Directory

#### 4.2.1 /checkpoints

This is the directory for saving checkpoints during training.

#### 4.2.2 /data ####

This is the directory for data labels and scripts generating them.

```
─retinaface_labels
     ├─test
     ├─train
     └─val
─convert_to_voc_format.py
─filter_labels.py
─wider_face_2_voc_add_landmark.py
```

#### 4.2.3 /jetcam ####

This is the directory for camera scripts.

#### 4.2.4 /models

Each sub-directory in /models contains a model parameter file and the corresponding labels.

```
models
    ├─onnx
    ├─pretrained
    ├─RFB-finetuned-age			# This is a model that tells if you are young or old
    ├─RFB-finetuned-emotion-320 #our main model
    └─train-version-RFB
```

#### 4.2.5 /vision 

This is the directory for storing model structure.

```
vision
    ├─datasets      
    ├─nn	
    ├─ssd		
    │  └─config
    ├─transforms
    └─utils
```

## 5.References ##

https://github.com/NVIDIA-AI-IOT/jetcam
https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB





