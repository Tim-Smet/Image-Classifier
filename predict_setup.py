import json
import torch
from torchvision import datasets, transforms, models
from PIL import Image
import numpy as np

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)

    if checkpoint['Model'] == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif checkpoint['Model'] == 'vgg19':
        model = models.vgg19(pretrained=True)
    else:
        print('Sorry I think you picked the wrong model.')
    
    model.classifier = checkpoint['Classifier']
    model.class_to_idx = checkpoint['Class_to_idx']  
    model.load_state_dict(checkpoint['State_dict'], strict=False)
    
    return model

def process_image(image_path):
    image = Image.open(image_path)

    if image.size[0] > image.size[1]:
        size = (50000, 256)
        image.thumbnail(size)
    else:
        size = (256, 50000)
        image.thumbnail(size)
    
    left_margin = (image.width-224)/2
    bottom_margin = (image.height-224)/2
    right_margin = left_margin + 224
    top_margin = bottom_margin + 224    
    image = image.crop((left_margin, bottom_margin, right_margin,   
                      top_margin))
    
    image = np.array(image)/255
    mean = np.array([0.485, 0.456, 0.406]) 
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean)/ std
    
    image = image.transpose((2, 0, 1))
    
    return image

def label_mapping(json_file):
    with open(json_file, 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name

def predict(image_path, model,json_file, mode, topk=5):
    device = torch.device(mode if torch.cuda.is_available() else "cpu")
    image = process_image(image_path)
    image_tensor = torch.from_numpy(image).type(torch.FloatTensor).float().to(device)
    model_input = image_tensor.unsqueeze(0)
    model.to(device)
    probs = torch.exp(model.forward(model_input))
    top_probs, top_labs = torch.topk(probs,topk)
    top_probs = top_probs.detach().cpu().numpy().tolist()[0]
    top_labs = top_labs.detach().cpu().numpy().tolist()[0]
    
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_labels = [idx_to_class[lab] for lab in top_labs]
    top_flowers = [label_mapping(json_file)[idx_to_class[lab]] for lab in top_labs]
    
    return top_probs, top_labels, top_flowers

