# inference.py
import matplotlib
matplotlib.use('Agg')
import torch
import os
import numpy as np
from models.model import FashionResnet
from data.dataset_fashion import get_attr_names_and_types
from data import input
from util import util
from torchvision.transforms import Compose, Resize, Normalize, ToTensor
from PIL import Image

def load_model(df_dir, checkpoint_path, resnet_type="resnet50"):
    model = FashionResnet(50, 1000, resnet_type)
    from train import load_model as load_ckpt
    load_ckpt(model, checkpoint_path, optimizer=None, devices=[])
    
    cat_names = input.get_ctg_name(df_dir)[0]
    cat_names = ['n/a'] + cat_names[0:-1]
    attr_names, attr_types = get_attr_names_and_types(df_dir)
    attr_names = np.asarray(attr_names)
    attr_types = torch.FloatTensor(np.asarray(attr_types))
    
    return model, cat_names, attr_names, attr_types

def predict(model, cat_names, attr_names, attr_types, image_path):
    transform = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        out_cls, out_bin, out_bbox = model(img)
        pred_class_idx = out_cls.argmax().item()
        category = cat_names[pred_class_idx]

        result = {
            "category": category,
            "attributes": []
        }

        for j in range(1, 6):
            out_bin_subset = out_bin.clone()
            out_bin_subset[:, (attr_types != j)] = -1000.0
            topk = out_bin_subset.topk(3, 1)
            indices = topk.indices[0].numpy()
            names = attr_names[indices]
            result["attributes"].append({
                "type": j,
                "values": names.tolist()
            })

        return result
