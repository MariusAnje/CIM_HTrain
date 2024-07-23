import torch
import torchvision
import torchvision.transforms as transforms

def get_loader(dataset, BS, shuffle, num_workers):
    return torch.utils.data.DataLoader(dataset, batch_size=BS, shuffle=shuffle, num_workers=num_workers)

def get_transforms(dataset_name):
    if dataset_name == "CIFAR10" or dataset_name == "CIFAR100":
        normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        transform = transforms.Compose(
        [transforms.ToTensor(),
            normalize])
        train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
                ])
    elif dataset_name == "TIN":
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        transform = transforms.Compose(
                [transforms.ToTensor(),
                 normalize,
                ])
        train_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(64, 4),
                transforms.ToTensor(),
                normalize,
                ])
    elif dataset_name == "ImageNet":
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        pre_process = [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
        ]
        pre_process += [
            transforms.ToTensor(),
            normalize
        ]
        train_transform = transforms.Compose(pre_process)
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize
            ])
    elif dataset_name == "MNIST":
        transform = transforms.ToTensor()
        train_transform = transforms.ToTensor()
    else:
        raise NotImplementedError(f"Data set {dataset_name} not implemented")
    return transform, train_transform

def get_dataset_object(dataset_name):
    transform, train_transform = get_transforms(dataset_name)
    if dataset_name == "CIFAR10":
        trainset = torchvision.datasets.CIFAR10(root='~/Private/data', train=True, download=False, transform=train_transform)
        testset = torchvision.datasets.CIFAR10(root='~/Private/data', train=False, download=False, transform=transform)
    elif dataset_name == "CIFAR100":
        trainset = torchvision.datasets.CIFAR100(root='~/Private/data', train=True, download=False, transform=train_transform)
        testset = torchvision.datasets.CIFAR100(root='~/Private/data', train=False, download=False, transform=transform)
    elif dataset_name == "TIN":
        trainset = torchvision.datasets.ImageFolder(root='~/Private/data/tiny-imagenet-200/train', transform=train_transform)
        testset = torchvision.datasets.ImageFolder(root='~/Private/data/tiny-imagenet-200/val',  transform=transform)
    elif dataset_name == "ImageNet":
        trainset = torchvision.datasets.ImageFolder('/data/data/share/imagenet/train',
                                transform=train_transform)
        testset = torchvision.datasets.ImageFolder('/data/data/share/imagenet/val',
                                transform=transform)
    elif dataset_name == "MNIST":
        trainset = torchvision.datasets.MNIST(root='~/Private/data', train=True,
                                                download=False, transform=train_transform)
        testset = torchvision.datasets.MNIST(root='~/Private/data', train=False,
                                            download=False, transform=transform)
    else:
        raise NotImplementedError(f"Data set {dataset_name} not implemented")
    return trainset, testset

def get_dataset(args, BS, NW):
    if args.model == "CIFAR" or args.model == "Res18" or args.model == "QCIFAR" or args.model == "QRes18" or args.model == "QDENSE":
        trainset, testset = get_dataset_object("CIFAR10")
    elif args.model == "QCIFAR100" or args.model == "QResC100":
        trainset, testset = get_dataset_object("CIFAR100")
    elif args.model == "TIN" or args.model == "QTIN" or args.model == "QVGG":
        trainset, testset = get_dataset_object("TIN")
    elif args.model == "QVGGIN" or args.model == "QResIN":
        trainset, testset = get_dataset_object("ImageNet")
    else:
        trainset, testset = get_dataset_object("MNIST")
    trainloader = get_loader(trainset, BS, True, NW)
    testloader = get_loader(testset, BS, False, NW)
    return trainloader, testloader

def get_split_dataset(dataset, ratio):
    first_size = int(ratio * len(dataset))
    second_size = len(dataset) - first_size
    first_dataset, second_dataset = torch.utils.data.random_split(dataset, [first_size, second_size])
    return first_dataset, second_dataset

def get_continue_dataset(args, BS, NW, ratio):
    if args.model == "CIFAR" or args.model == "Res18" or args.model == "QCIFAR" or args.model == "QRes18" or args.model == "QDENSE":
        trainset, testset = get_dataset_object("CIFAR10")
    elif args.model == "QCIFAR100" or args.model == "QResC100":
        trainset, testset = get_dataset_object("CIFAR100")
    elif args.model == "TIN" or args.model == "QTIN" or args.model == "QVGG":
        trainset, testset = get_dataset_object("TIN")
    elif args.model == "QVGGIN" or args.model == "QResIN":
        trainset, testset = get_dataset_object("ImageNet")
    else:
        trainset, testset = get_dataset_object("MNIST")
    trainset, continueset = get_split_dataset(trainset, ratio)
    trainloader = get_loader(trainset, BS, True, NW)
    countinueloader = get_loader(continueset, BS, True, NW)
    testloader = get_loader(testset, BS, False, NW)
    return trainloader, countinueloader, testloader


if __name__ == "__main__":
    class LOL():
        def __init__(self) -> None:
            self.model = "QVGGIN"
    
    def eval_dataset(model_name, args):
        args.model = model_name
        trainloader, testloader = get_dataset(args, 16, 2)
        print(next(iter(trainloader))[0].shape)
        print(next(iter(testloader))[0].shape)

    args = LOL()
    eval_dataset("CIFAR", args)
    eval_dataset("QCIFAR100", args)
    eval_dataset("TIN", args)
    eval_dataset("sdfsdf", args)
    