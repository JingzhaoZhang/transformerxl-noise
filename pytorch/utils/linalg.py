import time
import torch
import torch.autograd as autograd

def eigen_variance(net, criterion, dataloader, n_iters=10, tol=1e-2, verbose=False):
    n_parameters = num_parameters(net)
    v0 = torch.randn(n_parameters)

    Av_func = lambda v: variance_vec_prod(net, criterion, dataloader, v)
    mu = power_method(v0, Av_func, n_iters, tol, verbose)
    return mu


def eigen_hessian(net, dataloader, batches, mems, n_iters=10, tol=1e-2, verbose=False):
    n_parameters = num_parameters(net)
    v0 = torch.randn(n_parameters)

    Av_func = lambda v, mems: hessian_vec_prod(net, dataloader, batches, mems, v)
    mu, mems = power_method(v0, Av_func, mems, n_iters, tol, verbose)
    return mu, mems


def variance_vec_prod(net, criterion, dataloader, v):
    X, y = dataloader.X, dataloader.y
    Av, Hv, n_samples = 0, 0, len(y)

    for i in range(n_samples):
        bx, by = X[i:i+1].cuda(), y[i:i+1].cuda()
        Hv_i = Hv_batch(net, criterion, bx, by, v)
        Av_i = Hv_batch(net, criterion, bx, by, Hv_i)
        Av += Av_i
        Hv += Hv_i
    Av /= n_samples
    Hv /= n_samples
    H2v = hessian_vec_prod(net, criterion, dataloader, Hv)
    return Av - H2v


def hessian_vec_prod(net, dataloader, batches, mems, v):
    Hv_t = 0
    n_batchs = batches #len(dataloader)
    for _ in range(n_batchs):
        bx, by, _ = dataloader.__next__()
        Hv, mems= Hv_batch(net, bx.cuda(), by.cuda(), mems, v)
        Hv_t += Hv

    return Hv_t/n_batchs, mems


def Hv_batch(net, batch_x, batch_y, mems, v):
    """
    Hessian vector multiplication
    """
    net.eval()
    ret = net(batch_x, batch_y, *mems)
    loss, mems = ret[0], ret[1:]

#     loss = criterion(logits, batch_y)
    loss = loss.float().mean().type_as(loss)

    grads = autograd.grad(loss, net.parameters(), create_graph=True, retain_graph=True)
    idx, res = 0, 0
    for grad_i in grads:
        ng = torch.numel(grad_i)
        v_i = v[idx:idx+ng].cuda()
        res += torch.dot(v_i, grad_i.view(-1))
        idx += ng

    Hv = autograd.grad(res, net.parameters())
    Hv = [t.data.cpu().view(-1) for t in Hv]
    Hv = torch.cat(Hv)
    return Hv, mems


def power_method(v0, Av_func, mems, n_iters=10, tol=1e-3, verbose=False):
    mu = 0
    v = v0/v0.norm()
    for i in range(n_iters):
        time_start = time.time()

        Av, mems = Av_func(v, mems)
        mu_pre = mu
        mu = torch.dot(Av,v).item()
        v = Av/Av.norm()

        if abs(mu-mu_pre)/abs(mu) < tol:
            break
        if verbose:
            print('%d-th step takes %.0f seconds, \t %.2e'%(i+1,time.time()-time_start,mu))
    return mu, mems


def num_parameters(net):
    """
    return the number of parameters for given model
    """
    n_parameters = 0
    for para in net.parameters():
        n_parameters += para.data.numel()

    return n_parameters




