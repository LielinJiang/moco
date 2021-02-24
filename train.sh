CUDA_VISIBLE_DEVICES=3 python main_moco.py -a resnet34 --lr 0.03 --batch-size 64 --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 /workspace/datasets/data/ILSVRC2012/

CUDA_VISIBLE_DEVICES=0 python main_lincls.py \
  -a resnet50 \
  --lr 30.0 \
  --batch-size 4 \
  --pretrained ./checkpoint_0214/checkpoint_0199.pth.tar  \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \


CUDA_VISIBLE_DEVICES=0 python main_lincls.py \
  -a resnet50 \
  --lr 30.0 \
  --batch-size 4 \
  --pretrained /workspace/checkpoint_0199.pth.tar  \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \