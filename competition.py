############################################################################
### For a 2-nd year undergraduate student competition on
### the robustness of deep neural networks, where a student
### needs to develop
### 1. an attack algorithm, and
### 2. an adversarial training algorithm
###
### The score is based on both algorithms.
############################################################################


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchattacks
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.autograd import Variable
import argparse
import time
import copy

# input id
id_ = 201677259

# setup training parameters
parser = argparse.ArgumentParser(description='PyTorch MNIST Training')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--warmup_steps', type=int, default=4000,
                        help='warm up step size')
parser.add_argument('--gradient_accumulation_steps', default=4, type=int, required=False,
                        help='gradient accumulation')
parser.add_argument('--max_grad_norm', default=2.0, type=float,
                        required=False)

args = parser.parse_args(args=[])

# judge cuda is available or not
use_cuda = not args.no_cuda and torch.cuda.is_available()
# device = torch.device("cuda" if use_cuda else "cpu")
device = torch.device("cpu")

torch.manual_seed(args.seed)
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

train_set = torchvision.datasets.FashionMNIST(root='data', train=True, download=True,
                                              transform=transforms.Compose([transforms.ToTensor()]))
train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

test_set = torchvision.datasets.FashionMNIST(root='data', train=False, download=True,
                                             transform=transforms.Compose([transforms.ToTensor()]))
test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True)

# define fully connected network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        output = F.log_softmax(x, dim=1)
        return output
# generate adversarial data, you can define your adversarial method
def adv_attack(model, X, y, device):
    X_adv = Variable(X.data)
    # generate adversarial data using fgsm
    # random_noise = torch.FloatTensor(*X_adv.shape).uniform_(-0.1, 0.1).to(device)
    # X_adv = Variable(X_adv.data + random_noise)

    # PGD
    opt = optim.Adam([X_adv], lr=1e-3)
    opt.zero_grad()
    with torch.enable_grad():
        X = X_adv.to(device)
        y = y.to(device)
        loss = nn.CrossEntropyLoss()
        origin = X.data
        eps = 0.1099
        alpha = 2/255
        for i in range(20):
            X_adv.requires_grad = True
            outputs = model(X_adv)
            model.zero_grad()
            cost = loss(outputs, y).to(device)
            cost.backward()
            X_adv = X_adv + alpha * X_adv.grad.sign()
            # delta = torch.clamp(X_adv.data - X.data, min=-eps, max=eps)
            delta = torch.clamp(X_adv - origin, min=-eps, max=eps)
            X_adv = torch.clamp(origin + delta, min=0, max=1).detach_()
    return X_adv

def fgsm_attack(model, X, y, device):
    # fgsm
    X_adv = Variable(X.data)
    opt = optim.Adam([X_adv], lr=1e-3)
    opt.zero_grad()
    with torch.enable_grad():
        X_adv = X_adv.to(device)
        y = y.to(device)
        X_adv.requires_grad = True
        outputs = model(X_adv)
        model.zero_grad()
        loss = nn.CrossEntropyLoss()(outputs, y)
        loss.backward()
    eps = 0.5
    eta = eps*X_adv.grad.sign()
    X_adv = Variable(X_adv.data+eta, requires_grad=True)
    eta = torch.clamp(X_adv.data - X.data, -0.1099, 0.1099)
    X_adv = Variable(X.data+eta, requires_grad=True)

    return X_adv

def cw_attack(model, X, y, device):
    with torch.enable_grad():
        # X_adv = Variable(X.data)
        X_adv = 0.5*torch.log((1+X)/(1-X))
        X_adv = X_adv.to(device)
        X_adv.requires_grad = True
        MSELoss = nn.MSELoss(reduction='none')
        Flatten = nn.Flatten()
        optimizer = optim.Adam([X_adv], lr=1e-3)
        for step in range(50):
            X_adv = (1/2)*(torch.tanh(X_adv)+1)
            outputs = model(X_adv)
            model.zero_grad()
            current_L2 = MSELoss(Flatten(X), Flatten(X_adv)).sum(dim=1)
            L2_loss = current_L2.sum()
            f_loss = f(outputs, y).sum()
            cost = L2_loss + f_loss
            optimizer.zero_grad()
            cost.backward()
            optimizer.step()
            eta = torch.clamp(X_adv.data - X.data, -0.1099, 0.1099)
            X_adv = Variable(X.data + eta, requires_grad=True)

    return X_adv

def f(outputs, labels):
    one_hot_labels = torch.eye(len(outputs[0]))[labels].to(device)
    i, _ = torch.max((1 - one_hot_labels) * outputs, dim=1)  # get the second largest logit
    j = torch.masked_select(outputs, one_hot_labels.bool())  # get the largest logit
    return torch.clamp((i - j), min=-0.1099, max=0.1099)

def train(args, model, device, train_loader, optimizer, epoch, scheduler):
    # global random_noise
    step_size = 0.80  # Last step: 0.65
    data_, _ = next(iter(train_loader))

    eps = torch.zeros(args.batch_size, 28 * 28).to(device)
    eps.requires_grad = True

    data_ = torch.from_numpy(np.float32(data_))

    lr_steps = args.epochs * len(train_loader) * args.minibatch_replays

    model.train()
    n_repeat = 1

    for batch_idx, (data, target) in enumerate(train_loader):
        for j in range(n_repeat):
            for _ in range(args.minibatch_replays):
                data, target = data.to(device), target.to(device)
                data = data.view(data.size(0), 28 * 28)
                # random_noise = torch.FloatTensor(data.shape).uniform_(-0.11, 0.11).to(device)
                # random_noise.requires_grad=True
                # noise_batch = Variable(random_noise[0:data.size(0)], requires_grad=True)
                eps = Variable(eps[0:data.size(0)], requires_grad=True)
                model_data = data + eps
                # model_data = Variable(model_data[0:data.size(0)],requires_grad=True)

                # model_data.clamp_(-0.11,0.11)
                output = model(model_data)
                criterion = nn.CrossEntropyLoss()
                with torch.enable_grad():
                    loss = criterion(output, target)
                loss.backward()
                optimizer.zero_grad()
                # with torch.enable_grad():
                grad = eps.grad.detach()
                data_, _ = next(iter(train_loader))
                mu = torch.mean(data_)
                std = torch.std(data_)

                upper_lim = torch.tensor((1 - mu) / std)
                lower_lim = torch.tensor((0 - mu) / std)
                eps.data = torch.clamp(eps + 0.11 * torch.sign(grad), -0.11, 0.11)
                eps.data[:data.size(0)] = torch.clamp(eps[:data.size(0)], lower_lim-data, upper_lim-data)
                optimizer.step()
                eps.grad.zero_()

                scheduler.step()


        #use adverserial data to train the defense model
        adv_data= adv_attack(model, data, target, device=device)

        #clear gradients
        optimizer.zero_grad()

        #compute loss
        loss = F.nll_loss(model(adv_data), target)
        #loss = F.nll_loss(model(data), target)



        #get gradients and update
        loss.backward()
        optimizer.step()

# predict function
def eval_test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.view(data.size(0), 28 * 28)
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy


def eval_adv_test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.view(data.size(0), 28 * 28)
            adv_data = adv_attack(model, data, target, device=device)
            output = model(adv_data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy


# main function, train the dataset and print train loss, test loss for each epoch
def train_model():
    model = Net().to(device)
    # optimizer = optim.SGD(model.parameters(), lr=1e-3)
    # optimizer = optim.SGD(model.parameters(), lr=args.lr)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        # training
        train(args, model, device, train_loader, optimizer, epoch)

        # get trnloss and testloss
        trnloss, trnacc = eval_test(model, device, train_loader)
        advloss, advacc = eval_adv_test(model, device, train_loader)

        # print trnloss and testloss
        print('Epoch ' + str(epoch) + ': ' + str(int(time.time() - start_time)) + 's', end=', ')
        print('trn_loss: {:.4f}, trn_acc: {:.2f}%'.format(trnloss, 100. * trnacc), end=', ')
        print('adv_loss: {:.4f}, adv_acc: {:.2f}%'.format(advloss, 100. * advacc))

    adv_tstloss, adv_tstacc = eval_adv_test(model, device, test_loader)
    print('Your estimated attack ability, by applying your attack method on your own trained model, is: {:.4f}'.format(
        1 / adv_tstacc))
    print('Your estimated defence ability, by evaluating your own defence model over your attack, is: {:.4f}'.format(
        adv_tstacc))
    # save the model
    torch.save(model.state_dict(), str(id_) + '.pt')
    return model


# compute perturbation distance
def p_distance(model, train_loader, device):
    p = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data = data.view(data.size(0), 28 * 28)
        data_ = copy.deepcopy(data.data)
        adv_data = adv_attack(model, data, target, device=device)
        p.append(torch.norm(data_ - adv_data, float('inf')))
    print('epsilon p: ', max(p))


################################################################################################
## Note: below is for testing/debugging purpose, please comment them out in the submission file
################################################################################################

# Comment out the following command when you do not want to re-train the model
# In that case, it will load a pre-trained model you saved in train_model()
# model = train_model()

# Call adv_attack() method on a pre-trained model'
# the robustness of the model is evaluated against the infinite-norm distance measure
# important: MAKE SURE the infinite-norm distance (epsilon p) less than 0.11 !!!
# p_distance(model, train_loader, device)
