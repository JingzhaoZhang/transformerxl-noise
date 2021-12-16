# Transformer-XL:


This repo comes from https://github.com/kimiyoung/transformer-xl

It contains code for reproducing the transformer-xl result in "On Convergence of Training Loss Without Reaching Stationary Points"
https://arxiv.org/abs/2110.06256


## PyTorch

- Use getdata.sh to download wiki103

- Example usage:
```
cd pytorch

python train.py  --cuda  --data ../data/wikitext-103/ --dataset wt103 --adaptive  --n_layer 12  --d_model 410  --n_head 10  --d_head 41  --d_inner 2100  --dropout 0.1  --dropatt 0.0 --optim adam --lr 0.00025  --warmup_step 0  --max_step 200000  --tgt_len 150  --mem_len 150 --eval_tgt_len 150  --batch_size 60  --multi_gpu  --gpu0_bsz 4 --work_dir ../../logs/transformer -save_sharpness -save_noise   --save-dir test --scheduler constant
```
The above describes one particular plot for the constant step size training.
The stats needs to be smoothed to avoid periodicity in text. 

## Brief descriptions


noise and grad are computed in a straight forward way as described in https://arxiv.org/abs/2110.06256
Sharpness is computed using power method from https://github.com/leiwu0/sgd.stability
Output can be found in ./ckpt/

Unfortunately, I was not able to fix the code and support multigpu for pytorch >= 1.5

For detailed flag information, please refer to https://github.com/kimiyoung/transformer-xl
- save-dir - str, name of exp
- save_noise - bool, whether to compute and store noise/ grad norm
- noise_size - int, number of batch used to compute the full grad (sample number = noise_size x batch_size)
- noise_per_iter - int, number of train iterations per computation of noise
- save_sharpness - bool, whether to compute and store sharpness
- sharpness_batches - int, number of batch used to compute the sharpness (sample number = noise_size x batch_size)
- sharpness_per_iter - int, number of train iterations per computation of sharpness




