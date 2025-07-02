conda activate sfda
cd /remote-home/iot_hanxiang/ProSFDA
CUDA_VISIBLE_DEVICES=0 python main.py --cfg "cfgs/mypaper/gsource.yaml"
