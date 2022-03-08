#!/bin/bash
md=0.08
cuda_device=0
seed=0
env=Ant-v3
exp_name="tmap"
echo "$exp_name"
log_name="tmap_${md}_${seed}.log"
CUDA_VISIBLE_DEVICES=$cuda_device nohup python -u tmap.py --md ${md} --seed $seed --epoch 750 --env ${env} --exp_name $exp_name > $log_name &
	
