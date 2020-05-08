import argparse
import training_setup
parser = argparse.ArgumentParser(
    description='Hopefully this script helps in training the program ;-p',
)

parser.add_argument('--data_directory', dest='data_directory', action='store', default='./flowers')
parser.add_argument('--model_name', dest='model_name', action='store', default='vgg16')
parser.add_argument('--epochs', dest='epochs', action='store', default=5, type=int)
parser.add_argument('--learning_rate', dest='learning_rate', action='store', default=0.001, type=float)
parser.add_argument('--hidden_input', dest='hidden_input',  action='store', default=4096, type=int)
parser.add_argument('--save_dir', dest='save_dir', action='store', default='checkpoint.pth')
parser.add_argument('--cuda', dest="mode", action="store", default="cuda")

args = parser.parse_args()

trainset, trainloader, validloader, testloader = training_setup.load_data(args.data_directory)

model, criterion, optimizer = training_setup.create_model(args.model_name, args.hidden_input, args.learning_rate)

training_setup.train_model(model, criterion, optimizer, trainloader, validloader, args.epochs, args.mode)

training_setup.save_checkpoint(model, args, trainset, optimizer)