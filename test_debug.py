import torch
from metrics.STSIM_VGG import *

if __name__ == '__main__':
    device='cuda'
    opt = {"use_clip": True,"use_fourier": True,"fourier_dim":128,"use_color": True,"train":{"dropout_rate": 0.3,"layer_norm": True,"batch_norm": False}}
    model = STSIM_VGG([5900,10],opt,grayscale=False).to(device)
    # model.load_state_dict(torch.load('/home/pappas/STSIM/weights/STSIM_macro_VGG_05222022/epoch_0070.pt'), strict=False)
    model.to(device)
    tmp1 = torch.ones(1,3,128,128).to(device)
    tmp2 = torch.ones(2,3,128,128).to(device)
    # tmp2 = torch.zeros(10,5900).to(device)
    pred = model(tmp1, tmp2)
    print(pred)
    # import pdb;pdb.set_trace()