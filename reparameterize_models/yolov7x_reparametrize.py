"""reparmetrize yolov7x model, this repository will contain script to reparameterize the 
yolov7x version. 

The script needs model file and a configuration file for yolov7x
"""

from copy import deepcopy
from yolov7.models.yolo import Model
import torch
from yolov7.utils.torch_utils import select_device, is_parallel
import yaml


def model_yolov7x(model_file, model_config_file, save_reparametrized_model = ""):
    """convert yolov7x model to reparametrize model for deployment purposes
       and save the resultant model to a specified directory..

    Args:
        model_file: str -> yolov7x trained model file path 
        model_config_file: str -> yolov7x config file path (i.e. yolov7x.yaml) in deploy directory 
        saved_reparametrized_model: str -> path to  save the processed model file path default to ""
    """
    device = select_device('0', batch_size=1) # for single gpu
    ckpt = torch.load(model_file, map_location=device)
    model = Model(model_config_file, ch=3, nc=80).to(device)

    with open(model_config_file) as f:
        yml = yaml.load(f, Loader=yaml.SafeLoader)
    anchors = len(yml['anchors'][0]) // 2

    
    state_dict = ckpt['model'].float().state_dict()
    exclude = []
    intersect_state_dict = {k: v for k, v in state_dict.items() if k in model.state_dict() and not any(x in k for x in exclude) and v.shape == model.state_dict()[k].shape}
    model.load_state_dict(intersect_state_dict, strict=False)
    model.names = ckpt['model'].names
    model.nc = ckpt['model'].nc

    #reparametrize yolov7x
    for i in range((model.nc+5)*anchors):
        model.state_dict()['model.121.m.0.weight'].data[i, :, :, :] *= state_dict['model.121.im.0.implicit'].data[:, i, : :].squeeze()
        model.state_dict()['model.121.m.1.weight'].data[i, :, :, :] *= state_dict['model.121.im.1.implicit'].data[:, i, : :].squeeze()
        model.state_dict()['model.121.m.2.weight'].data[i, :, :, :] *= state_dict['model.121.im.2.implicit'].data[:, i, : :].squeeze()
    model.state_dict()['model.121.m.0.bias'].data += state_dict['model.121.m.0.weight'].mul(state_dict['model.121.ia.0.implicit']).sum(1).squeeze()
    model.state_dict()['model.121.m.1.bias'].data += state_dict['model.121.m.1.weight'].mul(state_dict['model.121.ia.1.implicit']).sum(1).squeeze()
    model.state_dict()['model.121.m.2.bias'].data += state_dict['model.121.m.2.weight'].mul(state_dict['model.121.ia.2.implicit']).sum(1).squeeze()
    model.state_dict()['model.121.m.0.bias'].data *= state_dict['model.121.im.0.implicit'].data.squeeze()
    model.state_dict()['model.121.m.1.bias'].data *= state_dict['model.121.im.1.implicit'].data.squeeze()
    model.state_dict()['model.121.m.2.bias'].data *= state_dict['model.121.im.2.implicit'].data.squeeze()

    
    ckpt = {'model': deepcopy(model.module if is_parallel(model) else model).half(),
            'optimizer': None,
            'training_results': None,
            'epoch': -1}

    # save reparameterized model
    torch.save(ckpt, save_reparametrized_model)
    


