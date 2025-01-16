import torch


def load_projected_weights(model:torch.nn.Module, projected_weights):
    print("----------------replace with new values----------------") #dev
    for name, params in model.named_parameters():
        prj_params = projected_weights.get(name)
        if prj_params is not None:
            print(name) #dev
            params.data.copy_(prj_params)
    print("--------------end------------------") #dev

    
