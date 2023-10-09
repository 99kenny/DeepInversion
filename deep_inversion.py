# --------------------------------------------------------
# Copyright (C) 2020 NVIDIA Corporation. All rights reserved.
# Nvidia Source Code License-NC
# Official PyTorch implementation of CVPR2020 paper
# Dreaming to Distill: Data-free Knowledge Transfer via DeepInversion
# Hongxu Yin, Pavlo Molchanov, Zhizhong Li, Jose M. Alvarez, Arun Mallya, Derek
# Hoiem, Niraj K. Jha, and Jan Kautz
# --------------------------------------------------------

import random
import collections
import torch
import torch.nn
import torch.optim as optim
import torch.cuda.amp as amp
import torchvision.utils as vutils
from PIL import Image 

from utils.utils import *

def get_image_prior_losses(inputs_jit):
    # b, c, h, w
    # difference right
    diff1 = inputs_jit[:,:,:,:-1] - inputs_jit[:,:,:,1:]
    # difference top
    diff2 = inputs_jit[:,:,:-1,:] - inputs_jit[:,:,1:,:]
    # difference top right
    diff3 = inputs_jit[:,:,1:,:-1] - inputs_jit[:,:,:-1,1:]
    # difference top bottom
    diff4 = inputs_jit[:,:,:-1,:-1] - inputs_jit[:,:,1:,1:]
    
    loss_var_l2 = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
    loss_var_l1 = (diff1.abs() / 255.0).mean() + (diff2.abs() / 255.0).mean() + (diff3.abs() / 255.0).mean() + (diff4.abs() / 255.0).mean()
    loss_var_l1 = loss_var_l1 * 255.0
    return loss_var_l1, loss_var_l2 

class DeepInversionFeatureHooK():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
    
    def hook_fn(self, module, input, output):
        nch = input[0].shape[1]
        mean = input[0].mean([0,2,3])
        var = input[0].permute(1,0,2,3).contiguous().view([nch, -1]).var(1, unbiased=False)
        
        r_feature = torch.norm(module.running_var.data - var, 2) + torch.norm(module.running_mean.data - mean, 2)
        self.r_feature = r_feature
    
    def close(self):
        self.hook.remove()
class DeepInversion(object):
    def __init__(self, class_num, 
                 net_teacher,
                 seed=0, bs=84, 
                 use_fp16=True,
                 path="./gen_images/",
                 final_data_path="./gen_images_final/",
                 setting_id=0,
                 jitter=30,
                 criterion=None,
                 hook_for_display=None,
                 
                 image_resolution = 224,
                 random_label = True,
                 start_noise = True,
                 detach_student = False,
                 do_flip = True,
                 store_best_images = True,
                    
                 bn_reg_scale = 0.05,
                 first_bn_multiplier = 10.,
                 var_scale_l1 = 0.0,
                 var_scale_l2 = 0.0001,
                 lr = 0.2,
                 l2 = 0.00001,
                 main_loss_multiplier = 1.0,
                 adi_scale = 0.0,
                 train_dataset=None,
                 ):
        
        torch.manual_seed(seed)
        self.net_teacher = net_teacher
        
        # params
        self.class_num = class_num
        self.image_resolution = image_resolution
        self.random_label = random_label
        self.start_noise = start_noise
        self.detach_student = detach_student
        self.do_flip = do_flip
        self.store_best_images = store_best_images
        
        self.setting_id = setting_id
        self.bs = bs
        self.use_fp16 = use_fp16
        self.save_every = 100
        self.jitter = jitter
        self.criterion = criterion
        do_clip = True
        # coefficients
        self.bn_reg_scale = bn_reg_scale
        self.first_bn_multiplier = first_bn_multiplier
        self.var_scale_l1 = var_scale_l1
        self.var_scale_l2 = var_scale_l2
        self.l2_scale = l2
        self.lr = lr
        self.main_loss_multiplier = main_loss_multiplier
        self.adi_scale = adi_scale
        
        self.num_generation = 0
        self.final_data_path = final_data_path
        self.best_path = path
        self.train_dataset = train_dataset
        create_folder(self.best_path)
        create_folder(self.final_data_path)
        
        self.loss_r_feature_layers = []
        
        for module in self.net_teacher.modules():
            if isinstance(module, nn.BatchNorm2d):
                self.loss_r_feature_layers.append(DeepInversionFeatureHooK(module))
                
        self.hook_for_display = hook_for_display
    
    def save_images(self, images, targets):
        # method to store generated images locally
        local_rank = torch.cuda.current_device()
        for id in range(images.shape[0]):
            class_id = targets[id].item()
            if 0:
                #save into separate folders
                place_to_store = f'{self.final_data_path}/s{class_id:03d}/img_{self.num_generation:05d}_id{id:03d}_gpu_{local_rank}_2.jpg'
            else:
                place_to_store = f'{self.final_data_path}/{class_id}_{self.num_generation}_id{id:03d}.jpg'

            image_np = images[id].data.cpu().numpy().transpose((1, 2, 0))
            pil_image = Image.fromarray((image_np * 255).astype(np.uint8))
            pil_image.save(place_to_store)

       
    def get_images(self, net_student, targets):
        class_num = self.class_num
        net_teacher = self.net_teacher
        use_fp16 = self.use_fp16
        save_every = self.save_every
        kl_loss =nn.KLDivLoss(reduction='batchmean').cuda()
        best_cost = 1e4
        criterion = self.criterion
        image_resolution = self.image_resolution
        data_type = torch.half if use_fp16 else torch.float
        train_dataset = self.train_dataset
        pooling_function = nn.modules.pooling.AvgPool2d(kernel_size=2)
        
        if targets is None or self.random_label :
            targets = torch.LongTensor([random.randint(0,class_num) for _ in range(self.bs)]).to('cuda')
                
        if train_dataset is None:
            inputs = torch.randn((self.bs, 3, image_resolution, image_resolution), requires_grad=True, device='cuda', dtype=data_type)
        else:
            inputs = torch.Tensor().cuda()
            # choose target image
            for index, target in enumerate(targets):
                idx = []
                for i in train_dataset.targets:
                    if i == target:
                        idx.append(i)
                inputs = torch.cat((inputs,torch.unsqueeze(train_dataset.__getitem__(random.choice(range(len(idx))))[0],0))).cuda()
                inputs.requires_grad =True
                
        targets = targets.type(torch.LongTensor).to('cuda')

        
        # multi resolution
        if self.setting_id == 0:
            skipFirst = False
        else:
            skipFirst = True
        print("targets", targets)
        iteration = 0
        for lr_it, lower_res in enumerate([2,1]):
            if lr_it == 0:
                iterations_per_layer = 2000
            else:
                iterations_per_layer = 1000 if not skipFirst else 2000
            
            if lr_it == 0 and skipFirst:
                continue
        
            lim_0, lim_1 = self.jitter // lower_res, self.jitter // lower_res
            
            if self.setting_id == 0:
                optimizer = optim.Adam([inputs], lr=self.lr, betas=[0.5,0.9], eps=1e-8)
                do_clip = True
            elif self.setting_id == 1:
                optimizer = optim.Adam([inputs], lr=self.lr, betas=[0.5, 0.9], eps=1e-8)
                do_clip = True
                
            if use_fp16:
                static_loss_scale = 256
                static_loss_scale = "dynamic"
                _, optimizer = amp.initialize([], optimizer, opt_level="O2", loss_scale=static_loss_scale)
            
            lr_scheduler = lr_cosine_policy(self.lr, 100, iterations_per_layer)
            
            for iteration_loc in range(iterations_per_layer):
                iteration += 1
                lr_scheduler(optimizer, iteration_loc, iteration_loc)
                
                # downsampling
                if lower_res!=1:
                    inputs_jit = pooling_function(inputs)
                else:
                    inputs_jit = inputs
                
                # apply random jitter offsets
                off1 = random.randint(-lim_0, lim_0)
                off2 = random.randint(-lim_1, lim_1)
                inputs_jit = torch.roll(inputs_jit, shifts=(off1, off2), dims=(2,3))
                
                # Flipping
                flip = random.random() > 0.5
                if flip and self.do_flip :
                    inputs_jit = torch.flip(inputs_jit, dims=(3,))
                
                optimizer.zero_grad()
                net_teacher.zero_grad()
                
                outputs = net_teacher(inputs_jit)
                
                # losses
                # classification loss
                loss = criterion(outputs, targets)
                
                # image regularization term
                # R_{prior} 
                loss_var_l1, loss_var_l2 = get_image_prior_losses(inputs_jit)
                # l2
                loss_l2 = torch.norm(inputs_jit.view(self.bs, -1), dim=1).mean()
                
                # feature distribution regularization
                # alpha feature
                rescale = [self.first_bn_multiplier] + [1. for _ in range(len(self.loss_r_feature_layers))]

                # R_{feature}
                loss_r_feature = sum([mod.r_feature * rescale[idx] for (idx, mod) in enumerate(self.loss_r_feature_layers)])
                
                # adaptive inversion
                # R_{compete}
                loss_verifier_cig = torch.zeros(1)
                if self.detach_student:
                    outputs_student = net_student(inputs_jit).detech()
                else:
                    outputs_student = net_student(inputs_jit)
                
                T = 3.0
                if 1:
                    T = 3.0
                    # JS divergence
                    # student
                    P = nn.functional.softmax(outputs_student / T, dim=1)
                    # teacher
                    Q = nn.functional.softmax(outputs / T, dim=1)
                    M = 0.5 * (P + Q)
                    # why clamp??
                    P = torch.clamp(P, 0.01, 0.99)
                    Q = torch.clamp(Q, 0.01, 0.99)
                    M = torch.clamp(M, 0.01, 0.99)
                    eps = 0.0
                    loss_verifier_cig = 0.5 * kl_loss(torch.log(P + eps), M) + 0.5 * kl_loss(torch.log(Q + eps), M)
                    loss_verifier_cig = 1.0 - torch.clamp(loss_verifier_cig, 0.0, 1.0)
                    
                # combine losses
                loss_aux = self.var_scale_l2 * loss_var_l2 + \
                    self.var_scale_l1 * loss_var_l1 + \
                    self.bn_reg_scale * loss_r_feature + \
                    self.l2_scale * loss_l2 + \
                    self.adi_scale * loss_verifier_cig + \
                    self.main_loss_multiplier * loss   
                    
                loss = loss_aux
                
                if iteration % save_every == 0:
                    print(f"------------iteration {iteration}----------")
                    print("total loss", loss.item())
                    print("loss_r_feature", loss_r_feature.item())
                    print("main criterion", criterion(outputs, targets).item())
                    print('loss_verifier_cig', loss_verifier_cig.item())
                    if self.hook_for_display is not None:
                        self.hook_for_display(inputs, targets)
                
                if use_fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                    
                optimizer.step()
                # clip yields better performance
                if do_clip:
                    inputs.data = clip(inputs.data, use_fp16=use_fp16)
                
                if best_cost > loss.item() or iteration == 1:
                    best_inputs = inputs.data.clone()
                    best_cost = loss.item()
                
                if iteration % save_every == 0 and (save_every > 0):
                    vutils.save_image(inputs, 
                                    f'{self.best_path}/output_{(iteration//save_every):05d}.png',
                                    normalize = True, scale_each = True, nrow=int(10))
        
        if self.store_best_images:
            best_inputs = denormalize(best_inputs)
            self.save_images(best_inputs, targets)
        
        # to reduce memory consompution
        optimizer.state = collections.defaultdict(dict)
            
            
    def generate_batch(self, net_student, targets=None):
        net_teacher = self.net_teacher
        use_fp16 = self.use_fp16
        
        net_student.eval()
        if targets is not None:
            targets = torch.from_numpy(np.array(targets * (self.bs // len(targets))).squeeze()).cuda()
        if use_fp16:
            targets = targets.half()
        
        self.get_images(net_student, targets)
        net_teacher.eval()
        self.num_generation += 1