import argparse
import predict_setup
parser = argparse.ArgumentParser(
    description='Hopefully this script helps in predicting a good output ;-p',
)

parser.add_argument('--image_path', dest='image_path', action='store')
parser.add_argument('--checkpoint_path', dest='checkpoint_path', action='store', default='checkpoint.pth')
parser.add_argument('--top_k', dest='top_k', action='store', default=5, type=int)
parser.add_argument('--category_names', dest='category_names', action='store', default='cat_to_name.json')
parser.add_argument('--cuda', dest="mode", action="store", default="cuda")

args = parser.parse_args()

model = predict_setup.load_checkpoint(args.checkpoint_path)

top_probs, top_labels, top_flowers = predict_setup.predict(args.image_path, model, args.category_names, args.mode, args.top_k)
dictionary = dict(zip(top_flowers, top_probs))
for key, val in dictionary.items():
    print("Flower: {} , Porbability: {}".format(key, val))
