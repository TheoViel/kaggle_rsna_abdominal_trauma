export OMP_NUM_THREADS=1
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 
CUDA_VISIBLE_DEVICES=0


cd src

#torchrun --nproc_per_node=8 main_seg_cls.py
torchrun --nproc_per_node=1 main_seg_cls.py
