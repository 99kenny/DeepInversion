#!/bin/sh
echo "enter root directory :"
read ROOT

# b
python3 $ROOT/main.py --path=$ROOT --knowledge_distillation --epochs=1 --jitter=0 --dataset=CIFAR10 --bs=100 --do_flip --exp_name=b --teacher=vgg11_bn --student=vgg11_bn --class_num=10 --image_resolution=32 --targets=0,1,2,3,4,5,6,7,8,9 --setting_id=1 --alpha_feature=1.0 --tv_l2=2.5e-5 --l2=3e-8 --adi_scale=0 --store_best_images
#c
python3 $ROOT/main.py --path=$ROOT --knowledge_distillation --epochs=1 --jitter=0 --dataset=CIFAR10 --bs=10 --do_flip --exp_name=c --teacher=vgg11_bn --student=vgg11_bn --class_num=10 --image_resolution=32 --targets=0,1,2,3,4,5,6,7,8,9 --setting_id=1 --alpha_feature=1.0 --tv_l2=2.5e-5 --l2=3e-8 --adi_scale=0 --store_best_images
#f
python3 $ROOT/main.py --path=$ROOT --knowledge_distillation --epochs=1 --jitter=0 --dataset=CIFAR10 --bs=10 --do_flip --exp_name=f --teacher=vgg11_bn --student=vgg11_bn --class_num=10 --image_resolution=32 --targets=0,1,2,3,4,5,6,7,8,9 --setting_id=1 --alpha_feature=1.0 --tv_l2=2.5e-5 --l2=3e-8 --adi_scale=0 --store_best_images --from_training

#e
python3 $ROOT/main.py --path=$ROOT --knowledge_distillation --epochs=1 --jitter=0 --dataset=CIFAR10 --bs=100 --do_flip --exp_name=e --teacher=vgg11_bn --student=vgg11_bn --class_num=10 --image_resolution=32 --targets=0,1,2,3,4,5,6,7,8,9 --setting_id=1 --alpha_feature=1.0 --tv_l2=2.5e-5 --l2=3e-8 --adi_scale=0 --store_best_images --from_training
#d
python3 $ROOT/main.py --path=$ROOT --knowledge_distillation --epochs=10 --jitter=0 --dataset=CIFAR10 --bs=100 --do_flip --exp_name=d --teacher=vgg11_bn --student=vgg11_bn --class_num=10 --image_resolution=32 --targets=0,1,2,3,4,5,6,7,8,9 --setting_id=1 --alpha_feature=1.0 --tv_l2=2.5e-5 --l2=3e-8 --adi_scale=0 --store_best_images --from_training

#test
#test c
echo "----------------------------------------------------------------"
echo "test c" 
python3 $ROOT/train_with_distilled.py --dataset=CIFAR10 --root=$ROOT --class_num=10 --model_name=vgg11_bn --exp_name=c --epochs=200
echo "test c with knowledge distillatin"
python3 $ROOT/train_with_distilled.py --dataset=CIFAR10 --root=$ROOT --class_num=10 --model_name=vgg11_bn --exp_name=c --epochs=200 --knowledge_transfer
#test f
echo "----------------------------------------------------------------"
echo "test f" 
python3 $ROOT/train_with_distilled.py --dataset=CIFAR10 --root=$ROOT --class_num=10 --model_name=vgg11_bn --exp_name=f --epochs=200
echo "test f with knowledge distillatin"
python3 $ROOT/train_with_distilled.py --dataset=CIFAR10 --root=$ROOT --class_num=10 --model_name=vgg11_bn --exp_name=f --epochs=200 --knowledge_transfer

#test e
echo "----------------------------------------------------------------"
echo "test e" 
python3 $ROOT/train_with_distilled.py --dataset=CIFAR10 --root=$ROOT --class_num=10 --model_name=vgg11_bn --exp_name=e --epochs=200
echo "test e with knowledge distillatin"
python3 $ROOT/train_with_distilled.py --dataset=CIFAR10 --root=$ROOT --class_num=10 --model_name=vgg11_bn --exp_name=e --epochs=200 --knowledge_transfer

#test d
echo "----------------------------------------------------------------"
echo "test d" 
python3 $ROOT/train_with_distilled.py --dataset=CIFAR10 --root=$ROOT --class_num=10 --model_name=vgg11_bn --exp_name=d --epochs=200
echo "test d with knowledge distillatin"
python3 $ROOT/train_with_distilled.py --dataset=CIFAR10 --root=$ROOT --class_num=10 --model_name=vgg11_bn --exp_name=d --epochs=200 --knowledge_transfer

#test b
echo "----------------------------------------------------------------"
echo "test b" 
python3 $ROOT/train_with_distilled.py --dataset=CIFAR10 --root=$ROOT --class_num=10 --model_name=vgg11_bn --exp_name=b --epochs=200
echo "test b with knowledge distillatin"
python3 $ROOT/train_with_distilled.py --dataset=CIFAR10 --root=$ROOT --class_num=10 --model_name=vgg11_bn --exp_name=b --epochs=200 --knowledge_transfer

