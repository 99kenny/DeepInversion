import argparse
import torch
import torch.nn as nn
import torchvision.models as models
import torch.cuda.amp as amp
import requests

from models import vgg

def validate_one(input, target, model):
    def accuracy(output, target, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

    with torch.no_grad():
        output = model(input)
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
    print("Verifier accuracy: ", prec1.item())

def select_models(dataset, name):
    if dataset == 'ImageNet':
        model = models.__dict__[name](pretrained=True)
    elif dataset == 'CIFAR10':
        if name == 'vgg11_bn':
            features = [64, 64, 128, 128, 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
            model = torch.nn.DataParallel(vgg.VggNet(features))
            model.load_state_dict(torch.load('./path/vgg.pth'))
            
            
    else:
        raise ValueError('unknown dataset')
    
    return model

def run(args):
    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print('loading torchvision model for teacher with the name {}'.format(args.teacher))
    teacher = select_models(args.dataset, args.teacher).to('cuda')
    teacher.eval()
    if args.use_fp16:
        teacher, _ = amp.initialize(teacher, [], opt_level='O2')
            
    print('loading torchvision model for student with the name {}'.format(args.student))
    student = select_models(args.dataset, args.student).to('cuda')
    student.eval()
        
    if args.use_fp16:
        student, _ = amp.initialize(student, [], opt_level='O2')
        student.to(device)
    
    student.train()
    if args.use_fp16:
        for module in student.modules:
            if isinstance(module, nn.BatchNorm2d):
                module.eval().half()
    
    from deep_inversion import DeepInversion
    
    exp_name = args.exp_name
    adi_data_path = "{}/results/final_images/{}".format(args.path, exp_name)
    best_path = "{}/results/best_images/{}".format(args.path, exp_name)
    hook_for_display = lambda x,y: validate_one(x,y, student)
    criterion = nn.CrossEntropyLoss()
    DeepInversionEngine = DeepInversion(class_num=args.class_num,
                  net_teacher= teacher,
                  seed=args.seed,
                  bs=args.bs,
                  use_fp16=args.use_fp16,
                  path=best_path,
                  final_data_path=adi_data_path,
                  jitter=args.jitter,
                  criterion=criterion,
                  hook_for_display=hook_for_display,
                  image_resolution=args.image_resolution,
                  random_label=args.random_label,
                  start_noise=args.start_noise,
                  do_flip=args.do_flip,
                  store_best_images=args.store_best_images,
                  bn_reg_scale=args.alpha_feature,
                  first_bn_multiplier=args.first_bn_multiplier,
                  var_scale_l1= args.tv_l1,
                  var_scale_l2= args.tv_l2,
                  lr=args.lr,
                  main_loss_multiplier=args.main_loss_multiplier,
                  adi_scale=args.adi_scale,
                  setting_id=args.setting_id,
                )
    targets = None
    if args.targets is not None:
        targets = [eval(i) for i in args.targets.split(',')]
    DeepInversionEngine.generate_batch(net_student=student, targets=targets)
    # train simple model for accuracy test on distilled dataset
    
def main():
    return 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # bs, seed, net_teacher, fp_16, 
    parser.add_argument('--alpha_feature',type=float, default=0.05, help='coefficient for feature distribution regularization')
    parser.add_argument('--first_bn_multiplier', type=float, default=10., help='additional multiplier on dirst bn layer of R_feature')
    parser.add_argument('--tv_l1', type=float, default=0.0, help='coefficient for tv l2 loss')
    parser.add_argument('--tv_l2', type=float, default=0.0001, help='coefficient for tv l2 loss')
    parser.add_argument('--lr', type=float, default=0.2, help='learning rate for optimization')
    parser.add_argument('--l2', type=float, default=0.00001, help='l2 loss on the image')
    parser.add_argument('--main_loss_multiplier', type=float, default=1.0, help='coefficient for the main loss in optimization')
    parser.add_argument('--adi_scale', type=float, default=0.0, help='alpha compete')
    
    parser.add_argument('--class_num', type=int, help='total number of classes in the dataset')
    parser.add_argument('--store_best_images', action='store_true', help='save best images as separate files')
    parser.add_argument('--epochs', type=int, default=2000, help="iterations")
    parser.add_argument('--setting_id', type=int, default=0, help="multi resolution : 0, else")
    parser.add_argument('--bs', type=int, default=128, help='batch size')
    parser.add_argument('--jitter', type=int, default=30, help='jitter')
    parser.add_argument('--teacher', type=str, default='resnet50', help='teacher model')
    parser.add_argument('--student', type=str, default='mobilenet_v2', help="student model")
    parser.add_argument('--use_fp16', action='store_true', help='use FP16 for optimization')
    parser.add_argument('--exp_name', type=str, default='test', help='where to store experimental data')
    parser.add_argument('--do_flip', action='store_true', help='apply flip during model inversion')
    parser.add_argument('--random_label', action='store_true', help='generate random label')
    parser.add_argument('--seed', type=int, default=0, help='seed')
    parser.add_argument('--start_noise', type=bool, default=True)
    parser.add_argument('--image_resolution', type=int, default=224, help='image resolution')
    parser.add_argument('--targets', type=str, help='targets')
    parser.add_argument('--dataset', type=str, default='ImageNet', help='dataset')
    parser.add_argument('--path', type=str, default='', help='results path')
    args = parser.parse_args()
    print(args)
    
    torch.backends.cudnn.benchmark = True
    run(args)
    
if __name__ == '__main__':
    main()