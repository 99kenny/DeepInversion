import argparse
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from datetime import datetime
import argparse
from models.multi_layer_cnn import MultiLayerCNN

from models import vgg
from utils.distilled_dataset import DistilledDataset

def train(train_loader, model, criterion, optimizer, epoch, writer, teacher):
    losses = 0.
    accs = 0.

    # switch to train mode
    model.train()

    for i, (input, target) in enumerate(train_loader):           
        target = target.cuda()
        input_var = input.cuda()
        soft_target = False
        if teacher is not None:
            soft_target = True
            target = teacher(input_var).softmax(dim=1)
        target_var = target

        # compute output
        output = model(input_var)
        # compute loss
        loss = criterion(output, target_var)

        # compute gradient and do SGD step
        optimizer.zero_grad()    # set gradients to zero
        loss.backward()          # compute gradients
        optimizer.step()         # step with learning rate

        output = output.float()
        loss = loss.float()

        # measure accuracy and record loss
        acc = accuracy(output.data, target, soft_target)[0]
        losses += loss.item()
        accs += acc
    accs /= len(train_loader)
    losses /= len(train_loader)
    print(f'[Epoch {epoch}] Average Loss : {losses:.3f}, Average Accuracy : {accs:.3f}')

    writer.add_scalar("Loss/train", losses, epoch)
    writer.add_scalar("Accuracy/train", accs, epoch)

def validate(val_loader, model, criterion, epoch, writer):
    losses = 0.
    accs = 0.

    # switch to evaluate mode
    model.eval()

    with torch.no_grad(): # disable tracking gradient to reduce memory use and increase computation speed
        for i, (input, target) in enumerate(val_loader):
            target = target.cuda()
            input_var = input.cuda()
            target_var = target.cuda()


            # compute output
            output = model(input_var)
            loss = criterion(output, target_var)
            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target, False)[0]
            losses += loss.item()
            accs += prec1.item()

        losses /= len(val_loader)
        accs /= len(val_loader)
        print(f'[Validation] : Average Loss {losses:.3f}, Average Accuracy {accs:.3f}')

        writer.add_scalar("Loss/val", losses, epoch)
        writer.add_scalar("Accuracy/val", accs, epoch)

    return accs

# top k accuacry
def accuracy(output, target, soft_target,topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    if soft_target:
        _, target = target.topk(1, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def train_with_distilled(dataset_name, class_num, root, exp_name, epochs, model_name='vgg11_bn', knowledge_transfer=False):
    # model
    if model_name == 'vgg11_bn':
        features = [64, 64, 128, 128, 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
        model = torch.nn.DataParallel(vgg.VggNet(features))
    elif model_name == 'three_layer_cnn':
        model = MultiLayerCNN(class_num)
    else:
        model = models.__dict__[model_name]()
    teacher = None
    if knowledge_transfer:
        teacher = torch.nn.DataParallel(vgg.VggNet(features))
        teacher.load_state_dict(torch.load(f'{root}/path/vgg.pth'))
        teacher.cuda()
    distilled_path = f"{root}/results/final_images/{exp_name}"
    # train set
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, 4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225]),
    ])
    train_data = DistilledDataset(distilled_path, transform=train_transform)
    train_loader = DataLoader(
        train_data,
        batch_size=128,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    dataset_path = f"{root}/data"
    # test set
    test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225]),
    ])

    if dataset_name == 'CIFAR10':
        test_data = datasets.CIFAR10(root=dataset_path, 
                                train=False, 
                                download=True, 
                                transform=test_transform)
    elif dataset_name == 'ImageNet':
        test_data = datasets.ImageNet(root=dataset_path, 
                                      split='val', 
                                      download=True, 
                                      transform=test_transform)
    else:
        raise ValueError('no such dataset')

    val_loader = DataLoader(
        test_data,
        batch_size=128,
        shuffle=False,
        num_workers=2,
        pin_memory=True)
    
    log_path = f"{root}/logs/{dataset_name}/{exp_name}/{model_name}"
    # writer
    writer = SummaryWriter(log_path, filename_suffix=datetime.now().strftime('%Y%m%d-%H%M'))
    
    # train
    lr = 0.01
    model.cuda()
    criterion = nn.CrossEntropyLoss(label_smoothing=0).cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr, momentum=0.9, weight_decay=5e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0)
    
    best_prec1 = 0.
    for epoch in range(0, epochs):
        print(f"current lr {optimizer.param_groups[0]['lr']:.5e}")

        train(train_loader, model, criterion, optimizer, epoch, writer, teacher)
        
        lr_scheduler.step()
        prec1 = validate(val_loader, model, criterion, epoch, writer)
        
        is_best = prec1 > best_prec1
        if is_best:
            torch.save(model.state_dict(), 'best_path')
            best_prec1 = prec1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--epochs', type=int, default=10, help='epochs')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--root', type=str, default='./', help='project root')
    parser.add_argument('--class_num', type=int, help='number of class')
    parser.add_argument('--model_name', type=str, help='model name to train : vgg11_bn, three_layer_cnn, ...')
    parser.add_argument('--exp_name', type=str, help='exp name')
    parser.add_argument('--knowledge_transfer', action='store_true', help='Use knowledge distillation to train student')
    args = parser.parse_args()
    print(args)
    
    train_with_distilled(args.dataset, 
                         args.class_num, 
                         args.root, 
                         args.exp_name, 
                         args.epochs, 
                         args.model_name,
                         args.knowledge_transfer)
    