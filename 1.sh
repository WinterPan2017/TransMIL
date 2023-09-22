CUDA_VISIBLE_DEVICES=4 python train.py --stage='train' --config='Camelyon/TransMIL.yaml'  --gpus=0 --fold=1 --save_dir=/home/pwt/MIL/PosMIL/data/posmil_logs/transmil/transmil_5fold/transmil_res50_1
CUDA_VISIBLE_DEVICES=4 python train.py --stage='test' --config='Camelyon/TransMIL.yaml'  --gpus=0 --fold=1 --save_dir=/home/pwt/MIL/PosMIL/data/posmil_logs/transmil/transmil_5fold/transmil_res50_1

CUDA_VISIBLE_DEVICES=4 python train.py --stage='train' --config='Camelyon/TransMIL.yaml'  --gpus=0 --fold=2 --save_dir=/home/pwt/MIL/PosMIL/data/posmil_logs/transmil/transmil_5fold/transmil_res50_2
CUDA_VISIBLE_DEVICES=4 python train.py --stage='test' --config='Camelyon/TransMIL.yaml'  --gpus=0 --fold=2 --save_dir=/home/pwt/MIL/PosMIL/data/posmil_logs/transmil/transmil_5fold/transmil_res50_2

CUDA_VISIBLE_DEVICES=4 python train.py --stage='train' --config='Camelyon/TransMIL.yaml'  --gpus=0 --fold=3 --save_dir=/home/pwt/MIL/PosMIL/data/posmil_logs/transmil/transmil_5fold/transmil_res50_3
CUDA_VISIBLE_DEVICES=4 python train.py --stage='test' --config='Camelyon/TransMIL.yaml'  --gpus=0 --fold=3 --save_dir=/home/pwt/MIL/PosMIL/data/posmil_logs/transmil/transmil_5fold/transmil_res50_3

CUDA_VISIBLE_DEVICES=4 python train.py --stage='train' --config='Camelyon/TransMIL.yaml'  --gpus=0 --fold=4 --save_dir=/home/pwt/MIL/PosMIL/data/posmil_logs/transmil/transmil_5fold/transmil_res50_4
CUDA_VISIBLE_DEVICES=4 python train.py --stage='test' --config='Camelyon/TransMIL.yaml'  --gpus=0 --fold=4 --save_dir=/home/pwt/MIL/PosMIL/data/posmil_logs/transmil/transmil_5fold/transmil_res50_4

CUDA_VISIBLE_DEVICES=4 python train.py --stage='train' --config='Camelyon/TransMIL.yaml'  --gpus=0 --fold=5 --save_dir=/home/pwt/MIL/PosMIL/data/posmil_logs/transmil/transmil_5fold/transmil_res50_5
CUDA_VISIBLE_DEVICES=4 python train.py --stage='test' --config='Camelyon/TransMIL.yaml'  --gpus=0 --fold=5 --save_dir=/home/pwt/MIL/PosMIL/data/posmil_logs/transmil/transmil_5fold/transmil_res50_5