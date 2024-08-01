import torch
import time
import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch import nn
import modules
from models import MLP3, LeNet, CIFAR
from torch import optim
import logging
from tqdm import tqdm
import resnet
from data import get_dataset, get_continue_dataset
import random

def get_model(args):
    if not hasattr(args, "device_type"):
        args.device_type = "RRAM1"
    if args.model == "MLP3":
        model = MLP3(args.device_type)
#     elif args.model == "MLP3_2":
#         model = SMLP3(args.device_type)
#     elif args.model == "MLP4":
#         model = SMLP4(args.device_type)
    elif args.model == "LeNet":
        model = LeNet(args.device_type)
    elif args.model == "CIFAR":
        model = CIFAR(args.device_type)
    elif args.model == "Res18":
        model = resnet.resnet18(num_classes = 10, args=args)
#     elif args.model == "TIN":
#         model = resnet.resnet18(num_classes = 200)
#     elif args.model == "QLeNet":
#         model = QSLeNet(args.device_type)
#     elif args.model == "QCIFAR":
#         model = QCIFAR(args.device_type)
#     elif args.model == "QCIFAR100":
#         model = QCIFAR100(args.device_type)
#     elif args.model == "QRes18":
#         model = qresnet.resnet18(num_classes = 10)
#     elif args.model == "QResC100":
#         model = qresnet.resnet18(num_classes = 100)
#     elif args.model == "QDENSE":
#         model = qdensnet.densenet121(num_classes = 10)
#     elif args.model == "QTIN":
#         model = qresnet.resnet18(num_classes = 200)
#     elif args.model == "QVGG":
#         model = qvgg.vgg16(num_classes = 1000)
#     elif args.model == "Adv":
#         model = SAdvNet(args.device_type)
#     elif args.model == "QVGGIN":
#         model = qvgg.vgg16(num_classes = 1000)
#     elif args.model == "QResIN":
#         model = qresnetIN.resnet18(num_classes = 1000)
    else:
        NotImplementedError
    return model

def prepare_model(model, device, args):
    model.to(device)
    if "TIN" in args.model or "Res" in args.model or "VGG" in args.model or "DENSE" in args.model:
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.train_epoch)
    else:
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [60])
    warm_optimizer = optim.SGD(model.parameters(), lr=1e-3)
    return model, optimizer, warm_optimizer, scheduler

def get_logger(filepath=None):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%m-%d %H:%M:%S'))
    logger.addHandler(console_handler)
    if filepath is not None:
        file_handler = logging.FileHandler(filepath+'.log', mode='w')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%m-%d %H:%M:%S'))
        logger.addHandler(file_handler)
    return logger

@torch.no_grad()
def UpdateBN(model_group):
    model, criteriaF, optimizer, scheduler, device, trainloader, testloader = model_group
    model.train()
    total = 0
    correct = 0
    
    with torch.no_grad():
        # for images, labels in tqdm(testloader, leave=False):
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            # images = images.view(-1, 784)
            outputs = model(images)

@torch.no_grad()
def CEval(model_group):
    model, criteriaF, optimizer, scheduler, device, trainloader, testloader = model_group
    model.eval()
    total = 0
    correct = 0

    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        # images = images.view(-1, 784)
        outputs = model(images)
        predictions = outputs.argmax(dim=1)
        correction = predictions == labels
        correct += correction.sum()
        total += len(correction)
    return (correct/total).cpu().item()

@torch.no_grad()
def UpRange(model_group):
    model, criteriaF, optimizer, scheduler, device, trainloader, testloader = model_group
    model.train()
    total = 0
    correct = 0

    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)
        # images = images.view(-1, 784)
        outputs = model(images)
    #     predictions = outputs.argmax(dim=1)
    #     correction = predictions == labels
    #     correct += correction.sum()
    #     total += len(correction)
    # return (correct/total).cpu().item()

@torch.no_grad()
def MEval(model_group, noise_type, dev_var, rate_max, rate_zero, write_var, **kwargs):
    model, criteriaF, optimizer, scheduler, device, trainloader, testloader = model_group
    model.eval()
    total = 0
    correct = 0
    model.clear_noise()
    
    model.set_noise_multiple(noise_type, dev_var, rate_max, rate_zero, write_var, **kwargs)
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        predictions = outputs.argmax(dim=1)
        correction = predictions == labels
        correct += correction.sum()
        total += len(correction)
    return (correct/total).cpu().item()

@torch.no_grad()
def MEachEval(model_group, noise_type, dev_var, rate_max, rate_zero, write_var, **kwargs):
    model, criteriaF, optimizer, scheduler, device, trainloader, testloader = model_group
    model.eval()
    total = 0
    correct = 0
    model.clear_noise()

    for images, labels in testloader:
        model.clear_noise()
        model.set_noise_multiple(noise_type, dev_var, rate_max, rate_zero, write_var, **kwargs)
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        predictions = outputs.argmax(dim=1)
        correction = predictions == labels
        correct += correction.sum()
        total += len(correction)
    return (correct/total).cpu().item()

def MTrain(model_group, epochs, header, noise_type, dev_var, rate_max, rate_zero, write_var, verbose=False, **kwargs):
    model, criteriaF, optimizer, scheduler, device, trainloader, testloader = model_group
    best_acc = 0.0
    set_noise = True
    for i in range(epochs):
        start_time = time.time()
        model.train()
        model.make_fast()
        running_loss = 0.
        # for images, labels in tqdm(trainloader):
        for images, labels in trainloader:
            model.clear_noise()
            if set_noise:
                model.set_noise_multiple(noise_type, dev_var, rate_max, rate_zero, write_var, **kwargs)
            optimizer.zero_grad()
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criteriaF(outputs,labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        # model.make_slow()
        test_acc = MEachEval(model_group, noise_type, dev_var, rate_max, rate_zero, write_var, **kwargs)
        model.clear_noise()
        noise_free_acc = CEval(model_group)
        if set_noise:
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(model.state_dict(), f"tmp_best_{header}.pt")
        if verbose:
            end_time = time.time()
            print(f"epoch: {i:-3d}, noisy acc: {test_acc:.4f}, clean acc: {noise_free_acc:.4f}, noise set: {set_noise}, loss: {running_loss / len(trainloader):.4f}, used time: {end_time - start_time:.4f}")
        scheduler.step()

def CTrain(model_group, epochs, header, verbose=False, **kwargs):
    model, criteriaF, optimizer, scheduler, device, trainloader, testloader = model_group
    best_acc = 0.0
    for i in range(epochs):
        start_time = time.time()
        model.train()
        model.make_fast()
        running_loss = 0.
        # for images, labels in tqdm(trainloader):
        for images, labels in trainloader:
            optimizer.zero_grad()
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criteriaF(outputs,labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        test_acc = CEval(model_group)
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), f"tmp_best_{header}.pt")
        if verbose:
            end_time = time.time()
            print(f"epoch: {i:-3d}, test acc: {test_acc:.4f}, best acc: {best_acc:.4f}, loss: {running_loss / len(trainloader):.4f}, used time: {end_time - start_time:.4f}")
        scheduler.step()

def str2bool(a):
    if a == "True":
        return True
    elif a == "False":
        return False
    else:
        raise NotImplementedError(f"{a}")

def fix_random_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False