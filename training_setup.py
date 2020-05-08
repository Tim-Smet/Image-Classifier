import json
import time
import torch
from collections import OrderedDict
from torch import nn
from torch import optim
from torchvision import models, datasets

def load_data(data_dir):
    from torchvision import transforms
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                          [0.229, 0.224, 0.225])])

    transforms = transforms.Compose([transforms.Resize(255),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406],
                                                        [0.229, 0.224, 0.225])])


    trainset = datasets.ImageFolder(train_dir, transform = train_transforms)
    validset = datasets.ImageFolder(valid_dir, transform = transforms)
    testset = datasets.ImageFolder(test_dir, transform = transforms)


    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
    validloader = torch.utils.data.DataLoader(validset, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)
    
    return trainset, trainloader, validloader, testloader

def label_mapping():
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    return cat_to_name
        
def create_model(model_name, hidden_input, learning_rate):
    cat_to_name = label_mapping()
    
    if model_name == 'vgg16':
        model = models.vgg16(pretrained=True)
        
    elif model_name == 'vgg19':
        model = models.vgg19(pretrained=True)
        
    else:
        print("You picked a model that is not available. Please choose vgg16 or resnet18\n")
        
    
        
    for param in model.parameters():
        param.requires_grad = False
        
    input_size = model.classifier[0].in_features
    output_size = len(cat_to_name)
    dropout_prob = 0.2
        
    model.classifier = nn.Sequential(OrderedDict([
                           ('fc1', nn.Linear(input_size, hidden_input)),
                           ('relu', nn.ReLU()),
                           ('dropout', nn.Dropout(dropout_prob)),
                           ('fc2', nn.Linear(hidden_input, output_size)),
                           ('output', nn.LogSoftmax(dim=1))]))
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    return model, criterion, optimizer

def train_model(model, criterion, optimizer, trainloader, validloader, epochs, mode):
    steps = 0
    device = torch.device(mode if torch.cuda.is_available() else "cpu")
    print("The model started training\n")
    
    train_losses, valid_losses = [], []
    for e in range(epochs):
        start = time.time()
        model.train()
        model.to(device)
        run_loss = 0
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
        
            log_ps = model.forward(images)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()
        
            run_loss += loss.item()
        
        else:
            valid_loss = 0
            accuracy = 0
        
            with torch.no_grad():
                model.eval()
                for images, labels in validloader:
                    images, labels = images.to(device), labels.to(device)
                    log_ps = model.forward(images)
                    valid_loss += criterion(log_ps, labels)
                
                    ps = torch.exp(log_ps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor))
        
            model.train()
        
            train_losses.append(run_loss/len(trainloader))
            valid_losses.append(valid_loss/len(validloader))

            print("Epoch: {}/{}.. ".format(e+1, epochs),
                  "Training Loss: {:.3f}.. ".format(train_losses[-1]),
                  "Validation Loss: {:.3f}.. ".format(valid_losses[-1]),
                  "Validation Accuracy: {:.3f}".format(accuracy/len(validloader)))
            
    print("Training is done\n")

def save_checkpoint(model, args, trainset, optimizer):
    model.class_to_idx = trainset.class_to_idx

    checkpoint = {'Epochs': args.epochs,
                      'Model': args.model_name,
                      'Classifier': model.classifier,
                      'State_dict': model.state_dict(),
                      'Optimizer_state_dict': optimizer.state_dict(),
                      'Class_to_idx': model.class_to_idx}
    torch.save(checkpoint, args.save_dir)
    print("checkpoint is saved \n")
      