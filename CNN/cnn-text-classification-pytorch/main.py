import random
import dill
from model import CNN_Text
from mydatasets import MR
import torch
import re
from train import train_model
from torchtext import data
import argparse

parser = argparse.ArgumentParser()

# model parameters
parser.add_argument("--output_dim", type=int, default=2, help="output_dim")
parser.add_argument("--channel_dim", type=int, default=1, help="channel_dim [default: 1]")
parser.add_argument("--kernel_sizes", type=str, default="[2,3,4,5,6,7,8,9]", help="")
parser.add_argument("--embed_dim", type=int, default=300, help="embed_dim[default: 300]")
parser.add_argument("--dropout_rate", type=float, default=0.5, help="DroputRate [default: 0.5]")

# train
parser.add_argument("--data_path", type=str, default='rt-polaritydata', help="data_path [default: rt-polaritydata]")
parser.add_argument("--epochs", type=int, default=10, help="nepochs [default: 10]")
parser.add_argument("--lr", type=int, default=1e-3, help="lr [deafult: 1e-3]")
parser.add_argument("--seed", type=int, default=123, help="Random Seed for reproductivity")
parser.add_argument("--batch_size", type=int, default=128, help="BatchSize [default: 128]")
parser.add_argument("--weight_decay", type=float, default=0.0, help="weight_decay [default: 0.0]")

# predict
parser.add_argument("--predict", type=str, default=None, help='The sentences you want to predict')

args = parser.parse_args()
epochs = args.epochs
kernel_sizes = list(map(int, args.kernel_sizes.strip("[").strip("]").split(',')))
embed_dim = args.embed_dim
output_dim = args.output_dim
dropout_rate = args.dropout_rate
data_path = args.data_path
channel_dim = args.channel_dim
seed = args.seed
lr = args.lr
batch_size = args.batch_size
weight_decay = args.weight_decay

def set_seed(seed):
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed(seed)  # gpu

    random.seed(seed)  # random and transforms
    torch.backends.cudnn.deterministic = True  # cudnn

def main_test(text):
    model = torch.load('model.pkl')
    with open("text.field", 'rb') as f:
        TEXT = dill.load(f)
    text = TEXT.preprocess(text)
    text = [TEXT.vocab.stoi[word] for word in text]
    x = torch.tensor([text], dtype=torch.long)
    model.eval()
    logits = model(x)
    pred = torch.argmax(logits, dim=1)[0]
    return "postive" if pred else "negative"

def main_train():
    def clean_str(string):
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string

    TEXT = data.Field(sequential=True, lower=True, batch_first=True)
    TEXT.preprocessing = data.Pipeline(clean_str)
    LABEL = data.Field(sequential=False, use_vocab=False, batch_first=True)

    trainset, valset = MR.splits(data_path, fields=[("text", TEXT), ("label", LABEL)])
    TEXT.build_vocab(trainset)

    with open("text.field", 'wb') as f:
        dill.dump(TEXT, f)

    trainiter = data.BucketIterator(trainset, batch_size=batch_size, sort_key=lambda x: len(x.text),
                                    shuffle=True, device=device)

    valiter = data.BucketIterator(valset, batch_size=batch_size, sort_key=lambda x: len(x.text),
                                  shuffle=True, device=device)

    model = CNN_Text(channel_dim, len(TEXT.vocab), embed_dim, output_dim, kernel_sizes, is_static=False,
                     dropout_rate=dropout_rate)
    model = model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr, weight_decay=weight_decay)
    train_model(epochs, model, trainiter, valiter, optimizer, criterion)

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Deivce: {}".format(device))
    # set_seed(seed)

    if args.predict is not None:
        ret = main_test(args.predict)
        print("Text: ", args.predict)
        print("Sentiment: ", ret)
    else:
        main_train()
