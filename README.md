# Transformer-XL:


This repo comes from https://github.com/kimiyoung/transformer-xl

It contains code for reproducing the transformer-xl result in "On Convergence of Training Loss Without Reaching Stationary Points"
https://arxiv.org/abs/2110.06256


## PyTorch

- Use getdata.sh to download wiki103
- Example usage:

python train.py  --cuda  --data ../data/wikitext-103/ --dataset wt103 --adaptive  --n_layer 12  --d_model 410  --n_head 10  --d_head 41  --d_inner 2100  --dropout 0.1  --dropatt 0.0 --optim adam --lr 0.00025  --warmup_step 0  --max_step 200000  --tgt_len 150  --mem_len 150 --eval_tgt_len 150  --batch_size 60  --multi_gpu  --gpu0_bsz 4 --work_dir /com_space/jingzhao/logs/transformer -save_sharpness -save_noise -noise_size 800 --save-dir 0727-largebatch-000025 --scheduler constant

## Brief descriptions






