import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torchvision
from torchvision import transforms
from torch.autograd import Variable
import argparse
import time
import copy


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


################################################################################################
## Note: below is the place you need to add your own attack algorithm
################################################################################################
def random_attack(model, X, y, device):
    X_adv = Variable(X.data)

    random_noise = torch.FloatTensor(*X_adv.shape).uniform_(-0.1, 0.1).to(device)
    X_adv = Variable(X_adv.data + random_noise)

    return X_adv


def adv_attack(model, X, y, device):
    X_adv = Variable(X.data)
    ################################################################################################
    ## Note: below is the place you need to edit to implement your own attack algorithm
    ################################################################################################
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

    # FGSM
    X_adv_f = Variable(origin)
    opt = optim.Adam([X_adv_f], lr=1e-3)
    opt.zero_grad()
    with torch.enable_grad():
        X_adv_f = X_adv_f.to(device)
        y = y.to(device)
        X_adv_f.requires_grad = True
        outputs = model(X_adv_f)
        model.zero_grad()
        loss = nn.CrossEntropyLoss()(outputs, y)
        loss.backward()
    eps = 0.5
    eta = eps * X_adv_f.grad.sign()
    X_adv_f = Variable(X_adv_f.data + eta, requires_grad=True)
    eta = torch.clamp(X_adv_f.data - X.data, -0.1099, 0.1099)
    X_adv_f = Variable(X.data + eta, requires_grad=True)

    X_adv = X_adv_f + 2 * X_adv
    eta = torch.clamp(X_adv - X.data, -0.1099, 0.1099)
    X_adv = Variable(X.data + eta, requires_grad=True)
    ################################################################################################
    ## end of attack method
    ################################################################################################

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
    one_hot_labels = torch.eye(len(outputs[0]))[labels]
    i, _ = torch.max((1 - one_hot_labels) * outputs, dim=1)  # get the second largest logit
    j = torch.masked_select(outputs, one_hot_labels.bool())  # get the largest logit
    return torch.clamp((i - j), min=-0.1099, max=0.1099)

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

def pgd_attack(model, X, y, device):
    X_adv = Variable(X.data)

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
################################################################################################
## end of attack method
################################################################################################

def eval_adv_test(model, device, test_loader, adv_attack_method):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.view(data.size(0), 28 * 28)
            adv_data = adv_attack_method(model, data, target, device=device)
            output = model(adv_data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    test_accuracy = correct / len(test_loader.dataset)
    return test_loss, test_accuracy


def evaluate_all_models(model_file, attack_method, test_loader, device):
    model = Net().to(device)
    model.load_state_dict(torch.load(model_file))

    adv_attack = attack_method
    ls, acc = eval_adv_test(model, device, test_loader, adv_attack)

    del model
    return 1 / acc, acc


def main():
    ################################################################################################
    ## Note: below is the place you need to load your own attack algorithm and defence model
    ################################################################################################

    # attack algorithms name, add your attack function name at the end of the list
    attack_method = [cw_attack, fgsm_attack, pgd_attack, adv_attack]

    # defense model name, add your attack function name at the end of the list
    model_file = ["/Users/martin/PycharmProjects/pythonProject/1000.pt", "/Users/martin/PycharmProjects/pythonProject/1001.pt", "/Users/martin/PycharmProjects/pythonProject/1000.pt", "/Users/martin/PycharmProjects/pythonProject/1.pt", "/Users/martin/PycharmProjects/pythonProject/1000_guo.pt"]

    ################################################################################################
    ## end of load
    ################################################################################################

    # number of attack algorithms
    num_all_attack = len(attack_method)
    # number of defense model
    num_all_model = len(model_file)

    # define the evaluation matrix number
    evaluation_matrix = np.zeros((num_all_attack, num_all_model))

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

    for i in range(num_all_attack):
        for j in range(num_all_model):
            attack_score, defence_score = evaluate_all_models(model_file[j], attack_method[i], test_loader, device)
            evaluation_matrix[i, j] = attack_score

    print("evaluation_matrix: ", evaluation_matrix)
    # Higher is better
    print("attack_score_mean: ", evaluation_matrix.mean(axis=1))
    # Higher is better
    print("defence_score_mean: ", 1 / evaluation_matrix.mean(axis=0))


main()