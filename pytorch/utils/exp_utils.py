import functools
import os, shutil

import numpy as np

import torch




def compute_noise(stoc_grads, true_grads):
    total_noise_sq = 0 
    total_grad_sq = 0
    for k in stoc_grads.keys():
        total_noise_sq += (stoc_grads[k]- true_grads[k]).norm(2).item() ** 2
        total_grad_sq += stoc_grads[k].norm(2).item() ** 2
    return total_noise_sq, total_grad_sq

def compute_norm(grads):
    total_grad_sq = 0
    for k in grads.keys():
        total_grad_sq += grads[k].norm(2).item() ** 2
    return total_grad_sq

def clone_grad(net, true_grads):
    for name, param in net.named_parameters():
        if param.grad is None:
            continue
        true_grads[name] = torch.clone(param.grad.data).detach()
        
def param_weights(net):
    weight_names = []
    weights = []
    for name, param in net.named_parameters():
        if param.requires_grad:
            weight_names.append(name)
            weights.append( param.norm(2).item())
    return weight_names, weights
        

def coord_noise(stoc_grads, true_grads):
    coordnoise = []
    for k in stoc_grads.keys():
        coordnoise.extend((stoc_grads[k]-
                           true_grads[k]).cpu().numpy().flatten().tolist()[:])
    coordnoise = np.array(coordnoise)
    return coordnoise


def logging(s, log_path, print_=True, log_=True):
    if print_:
        print(s)
    if log_:
        with open(log_path, 'a+') as f_log:
            f_log.write(s + '\n')


def get_logger(log_path, **kwargs):
    return functools.partial(logging, log_path=log_path, **kwargs)


def create_exp_dir(dir_path, scripts_to_save=None, debug=False):
    if debug:
        print('Debug Mode : no experiment dir created')
        return functools.partial(logging, log_path=None, log_=False)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    print('Experiment dir : {}'.format(dir_path))
    if scripts_to_save is not None:
        script_path = os.path.join(dir_path, 'scripts')
        if not os.path.exists(script_path):
            os.makedirs(script_path)
        for script in scripts_to_save:
            dst_file = os.path.join(dir_path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)

    return get_logger(log_path=os.path.join(dir_path, 'log.txt'))

def save_checkpoint(model, optimizer, path, epoch):
    torch.save(model, os.path.join(path, 'model_{}.pt'.format(epoch)))
    torch.save(optimizer.state_dict(), os.path.join(path, 'optimizer_{}.pt'.format(epoch)))
