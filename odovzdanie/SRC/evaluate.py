import torch
import argparse
from pathlib import Path
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from train_audio_gmm import AudioDataset, Pipeline, read_dataset, load_gmm, predict
from train_image_cnn import ConvNet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def audio_eval():
    t_gmm, nt_gmm = load_gmm()
    eval = read_dataset(AudioDataset("eval/audio"), Pipeline(), False)
    return predict(t_gmm, nt_gmm, eval)

def image_eval():
    eval_dataset = datasets.ImageFolder(root="eval/image", transform=v2.Compose([v2.ToTensor()]))
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False)
    image_classifier = ConvNet()
    image_classifier.load_state_dict(torch.load('image_classifier.pkl', map_location=torch.device(device)))
    image_classifier.eval()
    image_classifier.to(device)
    pred = []
    for i, eval_batch in enumerate(eval_loader):
        x, _ = eval_batch
        x = x.to(device)
        with torch.no_grad():
            y_hat = image_classifier(x)
            pred_labels = (torch.sigmoid(y_hat) > 0.2)
            pred.append({'name': eval_dataset.imgs[i][0], 'score': torch.sigmoid(y_hat).item(), 'target': pred_labels.int().item()})
    return pred

def print_predictions(pred):
    for p in pred:
        print("{} {:.2f} {}".format(Path(p['name']).stem, p['score'], p['target']))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", action="store_true")
    parser.add_argument("--image", action="store_true")
    args = parser.parse_args()

    if args.audio:
        print_predictions(audio_eval())
    elif args.image:
        print_predictions(image_eval())
    else:
        print("Please specify --audio or --image")