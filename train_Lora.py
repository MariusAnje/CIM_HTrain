import argparse
import torch
from utils import str2bool, fix_random_seed
from utils import get_dataset, get_logger, get_model, prepare_model, get_continue_dataset
from utils import MTrain, UpRange, CEval, MEachEval, CTrain
import Pmodels

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_epoch', action='store', type=int, default=20,
            help='# of epochs of training')
    parser.add_argument('--noise_epoch', action='store', type=int, default=100,
            help='# of epochs of noise validations')
    parser.add_argument('--train_var', action='store', type=float, default=0.1,
            help='device variation [std] when training')
    parser.add_argument('--dev_var', action='store', type=float, default=0.3,
            help='device variation [std] before write and verify')
    parser.add_argument('--write_var', action='store', type=float, default=0.03,
            help='device variation [std] after write and verify')
    parser.add_argument('--device_type', action='store', default="RRAM1",
            help='type of device, e.g., RRAM1')
    parser.add_argument('--compute_device', action='store', default="cuda:0",
            help='device used')
    parser.add_argument('--verbose', action='store', type=str2bool, default=False,
            help='see training process')
    parser.add_argument('--model', action='store', default="MLP3", choices=["MLP3", "MLP3_2", "MLP4", "LeNet", "CIFAR", "Res18", "TIN", "QLeNet", "QCIFAR", "QRes18", "QDENSE", "QTIN", "QVGG", "Adv", "QVGGIN", "QResIN"],
            help='model to use')
    parser.add_argument('--alpha', action='store', type=float, default=1e6,
            help='weight used in saliency - substract')
    parser.add_argument('--header', action='store',type=int, default=1,
            help='use which saved state dict')
    parser.add_argument('--seed', action='store',type=int, default=1,
            help='random seed to use')
    parser.add_argument('--pretrained', action='store',type=str2bool, default=True,
            help='if to use pretrained model')
    parser.add_argument('--model_path', action='store', default="./pretrained",
            help='where you put the pretrained model')
    parser.add_argument('--save_file', action='store',type=str2bool, default=True,
            help='if to save the files')
    parser.add_argument('--use_tqdm', action='store',type=str2bool, default=False,
            help='whether to use tqdm')
    args = parser.parse_args()

    print(args)

    fix_random_seed(args.seed)
    header = args.header

    BS = 128
    NW = 4
    trainloader, continueloader, testloader = get_continue_dataset(args, BS, NW, 0.2)
    # model = get_model(args)
    # model = Hmodels.CIFAR_Plain()
    # model = Hmodels.CIFAR_Res()
    model = Pmodels.CIFAR()
    model.make_fast()

    # device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    compute_device = torch.device(args.compute_device)
    model, optimizer, w_optimizer, scheduler = prepare_model(model, compute_device, args)
    criteria = torch.nn.CrossEntropyLoss()
    if args.pretrained:
        # state_dict = torch.load("saved_cifar10_0.1_0.2_reslike_246.pt",map_location=compute_device)
        state_dict = torch.load("saved_cifar10_0.1_0.2.pt",map_location=compute_device)
        model_dict = model.state_dict()
        filtered_state_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        model.load_state_dict(filtered_state_dict, strict=False)
        model.clear_noise()
        model.pre_training()
        model_group = model, criteria, optimizer, scheduler, compute_device, trainloader, testloader
        print("Pretrained Model.")
        # print(f"Fast: No mask no noise: {CEval(model_group):.4f}")
        # performance = MEachEval(model_group, "Four", args.train_var, 1, 1, 0, N=1, m=1)
        # print(f"Fast: No mask noise acc: {performance:.4f}")
    else:
        model.pre_training()
        model_group = model, criteria, optimizer, scheduler, compute_device, trainloader, testloader
        MTrain(model_group, args.train_epoch, header, "Four", args.train_var, 1, 1, 0, verbose=True, N=1, m=1)
        exit()
    
    # print(model.conv2.A[:, 0,0])
    # print(model.conv2.B[:, 0,0])
    # print(model.conv2.op.weight[0,0,:,:])
    model.Lo_only()
    Lo_parameters = model.get_Lo_parameters()
    optimizer = torch.optim.Adam(Lo_parameters, lr=1e-3)
    # optimizer = torch.optim.SGD(D_params, lr=1e-2, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [60])
    model.clear_noise()
    model.clear_mask()
    model.set_noise_multiple("Four", args.train_var, 1, 1, 0, N=1, m=1)
    model_group = model, criteria, optimizer, scheduler, compute_device, continueloader, testloader
    CTrain(model_group, args.train_epoch, header, verbose=True, N=1, m=1)
    # print(model.conv2.A[:, 0,0])
    # print(model.conv2.B[:, 0,0])
    # print(model.conv2.op.weight[0,0,:,:])

    state_dict = torch.load(f"tmp_best_{header}.pt")
    model.load_state_dict(state_dict)
#     model.clear_noise()
    print(f"Fast: No mask no noise: {CEval(model_group):.4f}")
    performance = MEachEval(model_group, "Four", args.train_var, 1, 1, 0, N=1, m=1)
    print(f"Fast: No mask noise acc: {performance:.4f}")