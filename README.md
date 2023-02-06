
# YOLOv7 Reparametrization

After training YOLOv7, you need to Reparametrize the checkpoints for deployment
purposes. Trained checkpoints may not convert to engine file for example when deploying yolov7 to Jetson. This repository organised yolov7 and its variants conversion from yolov7 trained checkpoints to Reparametrized checkpoints. Full Reparametrization deatial is avilable [here](https://arxiv.org/pdf/2207.02696.pdf)

![credit: yolov7](https://github.com/faizan1234567/YOLOv7-Reparameterization/blob/main/images/reparametrized_%20model.png)


## Installation

To install follow the procedure below;

```bash
 git clone https://github.com/faizan1234567/YOLOv7-Reparameterization
 cd YOLOv7-Reparameterization
```
Now run
```bash
pip install -r requirements.txt 
```
```bash
cd yolov7
pip install -r requirements.txt
```
follow yolov7 documentation for more installation instructions if needed.


## Usage

```python

-h, --help            show this help message and exit
  --weights WEIGHTS     path to the weight file of the model
  --model_config MODEL_CONFIG
                        path to the config file of a model
  --save SAVE           path to save the processed model checkpoints
  --yolov7_type YOLOV7_TYPE
                        yolov7 model to be convert i.e. yolov7, yolo7x, or
                     other variants..
```

