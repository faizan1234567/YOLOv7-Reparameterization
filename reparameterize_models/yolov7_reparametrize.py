"""reparmetrize yolov7 model, this repository will contain script to reparameterize the 
yolov7 version for deployment purposes. 

The script needs model file and a configuration file for yolov7
credit: https://github.com/WongKinYiu/yolov7/blob/main/tools/reparameterization.ipynb
"""
from copy import deepcopy
from yolov7.models.yolo import Model
import torch
from yolov7.utils.torch_utils import select_device, is_parallel
import yaml
import os


def model_yolov7(model_file, model_config_file, save_reparametrized_model = ""):
    """convert yolov7 model to reparametrize model for deployment purposes
       and save the resultant model to a specified directory..

    Args:
        model_file: str -> yolov7 trained model file path 
        model_config_file: str -> yolov7 config file path (i.e. yolov7.yaml) in deploy directory 
        saved_reparametrized_model: str -> path to  save the processed model file path default to ""
    """
    device = select_device('0', batch_size=1)
    ckpt = torch.load(model_file, map_location=device)
    # print(model_file)
    model = Model(model_config_file, ch=3, nc=5).to(device)

    with open(model_config_file) as f:
        yml = yaml.load(f, Loader=yaml.SafeLoader)
    anchors = len(yml['anchors'][0]) // 2


    state_dict = ckpt['model'].float().state_dict()
    print(state_dict.keys())
    exclude = []
    intersect_state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict() and not any(x in k for x in exclude) and v.shape == model.state_dict()[k].shape}
    model.load_state_dict(intersect_state_dict, strict=False)
    model.names = ckpt['model'].names
    model.nc = ckpt['model'].nc

    # reparametrized YOLOv7
    # reparametrized YOLOv7
    for i in range((model.nc+5)*anchors):
        model.state_dict()['model.105.m.0.weight'].data[i, :, :, :] *= state_dict['model.105.im.0.implicit'].data[:, i, : :].squeeze()
        model.state_dict()['model.105.m.1.weight'].data[i, :, :, :] *= state_dict['model.105.im.1.implicit'].data[:, i, : :].squeeze()
        model.state_dict()['model.105.m.2.weight'].data[i, :, :, :] *= state_dict['model.105.im.2.implicit'].data[:, i, : :].squeeze()
    model.state_dict()['model.105.m.0.bias'].data += state_dict['model.105.m.0.weight'].mul(state_dict['model.105.ia.0.implicit']).sum(1).squeeze()
    model.state_dict()['model.105.m.1.bias'].data += state_dict['model.105.m.1.weight'].mul(state_dict['model.105.ia.1.implicit']).sum(1).squeeze()
    model.state_dict()['model.105.m.2.bias'].data += state_dict['model.105.m.2.weight'].mul(state_dict['model.105.ia.2.implicit']).sum(1).squeeze()
    model.state_dict()['model.105.m.0.bias'].data *= state_dict['model.105.im.0.implicit'].data.squeeze()
    model.state_dict()['model.105.m.1.bias'].data *= state_dict['model.105.im.1.implicit'].data.squeeze()
    model.state_dict()['model.105.m.2.bias'].data *= state_dict['model.105.im.2.implicit'].data.squeeze()
        # model to be saved
    ckpt = {'model': deepcopy(model.module if is_parallel(model) else model).half(),
            'optimizer': None,
            'training_results': None,
            'epoch': -1}

    # save reparameterized model
    path = os.path.join(save_reparametrized_model, 'yolov7_reparameterized.pt')
    torch.save(ckpt, path)
    


