"""
    PyTorch training code for:
        "H-AT: Hybrid Attention Transfer for Knowledge Distillation"
        https://
    
    Origin codes forked from: 
        https://github.com/szagoruyko/attention-transfer (thanks to Sergey Zagoruyko et al. and their great work)

    This file includes:
        * CIFAR Wide ResNet training code (refer to https://github.com/szagoruyko/wide-residual-networks)
        * Hybrid Attention Transfer for Knowledge Distillation
        * Activation-based spatial attention transfer implementation
        * Knowledge distillation implementation
        * Similarity-preserving knowledge distillation implementation
    
    How to run this script:
        refer to cifar_train.sh

    2019 Yan Qu
"""


import argparse
import os
import json
import numpy as np
from tqdm import tqdm
import torch
from torch.optim import SGD
import torchvision.transforms as T
from torchvision import datasets
import torch.nn.functional as F
import torchnet as tnt
from torchnet.engine import Engine
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from nested_dict import nested_dict
from functools import partial
from torch.nn.init import kaiming_normal_
import numpy
import heapq


## KD
def knowledge_distillation(y, teacher_scores, labels, T, alpha):
    p = F.log_softmax(y/T, dim=1)
    q = F.softmax(teacher_scores/T, dim=1)
    l_kl = F.kl_div(p, q, size_average=False) * (T**2) / y.shape[0]
    l_ce = F.cross_entropy(y, labels)
    return l_kl * alpha + l_ce * (1. - alpha)

## AT

def at(x):
    return F.normalize(x.pow(2).mean(1).view(x.size(0), -1))


def at_loss(x, y):
    return (at(x) - at(y)).pow(2).mean()

## SP

def sp(x):
    Q = x.view(x.size(0), -1)
    G = F.normalize(torch.mm(Q, Q.permute(1, 0)))
    return G


def sp_loss(x, y):
    return (sp(x) - sp(y)).pow(2).mean()

## CSH (H-AT)

def to_embedding(x):
    return F.normalize(x.view(x.size(1), -1).view(x.size(0), -1))


def sim_loss(x, y):
    return (1 - (x * y).sum(1)/(x.pow(2).sum(1) * y.pow(2).sum(1))**0.5).pow(2).mean()


def n_std(x):
    std = torch.std(x, dim=0)
    n_std = std/(((std * std).sum())**0.5)
    return n_std


def top_n_index(x, n):
    a = x.cpu().numpy()
    top_n_index = heapq.nlargest(n, range(len(a)), a.take)
    top_n_index.sort()
    return top_n_index


def hybrid_loss(g_s, g_t, selected, params, mode, group_nums, bns):
    loss_group = []
    delta_muls = []
    beta_muls = []
    k=10**-group_nums
    for i in range(group_nums):
        k = k * 10
        delta_muls.append(k)
        beta_muls.append(100 * k)
    # print('delta_muls=', delta_muls, 'beta_muls=', beta_muls)
    for i in range(group_nums):
        ## get teacher embeddings in a batch to calc std
        e = to_embedding(F.avg_pool2d(F.relu(batch_norm(g_t[i], params, 'teacher.' + bns[i], mode)), g_t[i].size()[2], 1, 0))
        select = range(g_s[i].size()[1])
        ## if stu.width not equal to teacher.width, select the channel
        if g_s[i].size()[1] != g_t[i].size()[1]:
            if i > len(selected) - 1:
                ## select channel
                std = n_std(e)
                selected.append(top_n_index(std, g_s[i].size()[1]))
            if i == len(beta_muls) - 1:
                ## using the the selected channel
                select = selected[i]
            else:
                delta_muls[i] = 0
        # print('i, delta_muls[i], len(select) =', i, delta_muls[i], len(select))
        # print(selected)
        m_t = torch.index_select(g_t[i], 1, torch.LongTensor(select).cuda())
        e_t = torch.index_select(e, 1, torch.LongTensor(select).cuda())
        e_s = to_embedding(F.avg_pool2d(F.relu(batch_norm(g_s[i], params, 'student.' + bns[i], mode)), g_s[i].size()[2], 1, 0))
        if beta_muls[i] > 0:
            loss_group.append(beta_muls[i] * at_loss(g_s[i], m_t))
        if delta_muls[i] > 0:
            loss_group.append(delta_muls[i] * sim_loss(e_s, e_t))
    return loss_group


## param utils

def cast(params, dtype='float'):
    if isinstance(params, dict):
        return {k: cast(v, dtype) for k,v in params.items()}
    else:
        return getattr(params.cuda() if torch.cuda.is_available() else params, dtype)()


def conv_params(ni, no, k=1):
    return kaiming_normal_(torch.Tensor(no, ni, k, k))


def linear_params(ni, no):
    return {'weight': kaiming_normal_(torch.Tensor(no, ni)), 'bias': torch.zeros(no)}


def bnparams(n):
    return {'weight': torch.rand(n),
            'bias': torch.zeros(n),
            'running_mean': torch.zeros(n),
            'running_var': torch.ones(n)}

def flatten(params):
    return {'.'.join(k): v for k, v in nested_dict(params).items_flat() if v is not None}


def batch_norm(x, params, base, mode):
    return F.batch_norm(x, weight=params[base + '.weight'],
                        bias=params[base + '.bias'],
                        running_mean=params[base + '.running_mean'],
                        running_var=params[base + '.running_var'],
                        training=mode)


def print_tensor_dict(params):
    kmax = max(len(key) for key in params.keys())
    for i, (key, v) in enumerate(params.items()):
        print(str(i).ljust(5), key.ljust(kmax + 3), str(tuple(v.shape)).ljust(23), torch.typename(v), v.requires_grad)


def set_requires_grad_except_bn_(params):
    for k, v in params.items():
        if not k.endswith('running_mean') and not k.endswith('running_var'):
            v.requires_grad = True


## definition of WideResNet

def wide_resnet(depth, width, num_classes):
    assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
    n = (depth - 4) // 6
    widths = [int(v * width) for v in (16, 32, 64)]

    def gen_block_params(ni, no):
        return {
            'conv0': conv_params(ni, no, 3),
            'conv1': conv_params(no, no, 3),
            'bn0': bnparams(ni),
            'bn1': bnparams(no),
            'convdim': conv_params(ni, no, 1) if ni != no else None,
        }

    def gen_group_params(ni, no, count):
        return {'block%d' % i: gen_block_params(ni if i == 0 else no, no)
                for i in range(count)}

    flat_params = cast(flatten({
        'conv0': conv_params(3, 16, 3),
        'group0': gen_group_params(16, widths[0], n),
        'group1': gen_group_params(widths[0], widths[1], n),
        'group2': gen_group_params(widths[1], widths[2], n),
        'bn': bnparams(widths[2]),
        'fc': linear_params(widths[2], num_classes),
    }))

    set_requires_grad_except_bn_(flat_params)

    def block(x, params, base, mode, stride):
        o1 = F.relu(batch_norm(x, params, base + '.bn0', mode), inplace=True)
        y = F.conv2d(o1, params[base + '.conv0'], stride=stride, padding=1)
        o2 = F.relu(batch_norm(y, params, base + '.bn1', mode), inplace=True)
        z = F.conv2d(o2, params[base + '.conv1'], stride=1, padding=1)
        if base + '.convdim' in params:
            return z + F.conv2d(o1, params[base + '.convdim'], stride=stride)
        else:
            return z + x

    def group(o, params, base, mode, stride):
        for i in range(n):
            o = block(o, params, f'{base}.block{i}', mode, stride if i == 0 else 1)
        return o
    
    def f0(x, params, mode, base=''):
        return group(x, params, f'{base}group0', mode, 1)
    def f1(g0, params, mode, base=''):
        return group(g0, params, f'{base}group1', mode, 2)
    def f2(g1, params, mode, base=''):
        return group(g1, params, f'{base}group2', mode, 2)
    def f3(g2, params, mode, base=''):
        o = F.relu(batch_norm(g2, params, f'{base}bn', mode))
        o = F.avg_pool2d(o, 8, 1, 0)
        o = o.view(o.size(0), -1)
        return o
    
    def f(input, targets, params, mode, base=''):
        x = F.conv2d(input, params[f'{base}conv0'], padding=1)
        g0 = f0(x, params, mode, base)
        g1 = f1(g0, params, mode, base)
        g2 = f2(g1, params, mode, base)
        embedding = f3(g2, params, mode, base)
        o = F.linear(embedding, params[f'{base}fc.weight'], params[f'{base}fc.bias'])
        return o, (g0, g1, g2)

    return f, (f0, f1, f2, f3) , flat_params


## data pre-process

def create_dataset(opt, train):
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(np.array([125.3, 123.0, 113.9]) / 255.0,
                    np.array([63.0, 62.1, 66.7]) / 255.0),
    ])
    if train:
        transform = T.Compose([
            T.Pad(4, padding_mode='reflect'),
            T.RandomHorizontalFlip(),
            T.RandomCrop(32),
            transform
        ])
    return getattr(datasets, opt.dataset)(opt.dataroot, train=train, download=True, transform=transform)


cudnn.benchmark = True


## define input args

parser = argparse.ArgumentParser(description='Wide Residual Networks')
## model options
parser.add_argument('--depth', default=16, type=int)
parser.add_argument('--width', default=1, type=float)
parser.add_argument('--dataset', default='CIFAR10', type=str)
parser.add_argument('--dataroot', default='.', type=str)
parser.add_argument('--dtype', default='float', type=str)
parser.add_argument('--nthread', default=4, type=int)
parser.add_argument('--teacher_id', default='', type=str)
## training options
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--weight_decay', default=0.0005, type=float)
parser.add_argument('--epoch_step', default='[60,120,160]', type=str,
                    help='json list with epochs to drop lr on')
parser.add_argument('--lr_decay_ratio', default=0.2, type=float)
parser.add_argument('--randomcrop_pad', default=4, type=float)
parser.add_argument('--temperature', default=4, type=float)
parser.add_argument('--alpha', default=0, type=float)
parser.add_argument('--beta', default=0, type=float)
parser.add_argument('--gamma', default=0, type=float)
parser.add_argument('--delta', default=0, type=float)
## device options
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--save', default='', type=str,
                    help='save parameters and logs in this folder')
parser.add_argument('--gpu_id', default='0', type=str,
                    help='id(s) for CUDA_VISIBLE_DEVICES')


## main

def main():
    opt = parser.parse_args()
    print('parsed options:', vars(opt))
    epoch_step = json.loads(opt.epoch_step)
    num_classes = 10

    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_id

    def create_iterator(mode):
        return DataLoader(create_dataset(opt, mode), opt.batch_size, shuffle=mode,
                          num_workers=opt.nthread, pin_memory=torch.cuda.is_available())

    train_loader = create_iterator(True)
    test_loader = create_iterator(False)

    ## student
    f_s, f_s_part, params_s = wide_resnet(opt.depth, opt.width, num_classes)

    ## teacher
    if opt.teacher_id:
        selected = []
        with open(os.path.join('logs', opt.teacher_id, 'log.txt'), 'r') as ff:
            line = ff.readline()
            r = line.find('json_stats')
            info = json.loads(line[r + 12:])
        f_t, f_t_part, _ = wide_resnet(info['depth'], info['width'], num_classes)
        model_data = torch.load(os.path.join('logs', opt.teacher_id, 'model.pt7'))
        params_t = model_data['params']
        ## merge teacher and student params
        params = {'student.' + k: v for k, v in params_s.items()}
        for k, v in params_t.items():
            params['teacher.' + k] = v.detach().requires_grad_(False)
        
        def f(inputs, targets, params, mode):
            with torch.no_grad():
                y_t, g_t = f_t(inputs, targets, params, False, 'teacher.')
            y_s, g_s = f_s(inputs, targets, params, mode, 'student.')
            group_nums = len(g_s)
            bns= ['group1.block0.bn0', 'group2.block0.bn0', 'bn']
            at_losses = []
            sp_losses = []
            for i in range(group_nums):
                at_losses.append(at_loss(g_s[i], g_t[i]))
            sp_losses.append(sp_loss(g_s[group_nums - 1], g_t[group_nums - 1]))
            csh_losses = hybrid_loss(
                g_s=g_s,
                g_t=g_t,
                selected=selected,
                params=params,
                mode=mode, 
                group_nums=group_nums,
                bns=bns
            )
            return y_s, y_t, {'AT':at_losses, 'SP':sp_losses, 'CSH':csh_losses}
    else:
        f, params = f_s, params_s

    def create_optimizer(opt, lr):
        print('creating optimizer with lr = ', lr)
        ## only update stu params
        return SGD((v for v in params_s.values() if v.requires_grad), lr,
                   momentum=0.9, weight_decay=opt.weight_decay)

    optimizer = create_optimizer(opt, opt.lr)

    epoch = 0

    print('\nParameters:')
    print_tensor_dict(params)

    n_parameters = sum(p.numel() for p in list(params_s.values()))
    print('\nTotal number of parameters:', n_parameters)

    meter_loss = tnt.meter.AverageValueMeter()
    classacc = tnt.meter.ClassErrorMeter(accuracy=True)
    timer_train = tnt.meter.TimeMeter('s')
    timer_test = tnt.meter.TimeMeter('s')
    meters_distill = {}
    meters_distill['AT'] = [tnt.meter.AverageValueMeter() for i in range(3)]
    meters_distill['SP'] = [tnt.meter.AverageValueMeter()]
    meters_distill['CSH'] = [tnt.meter.AverageValueMeter() for i in range(6)]
    

    if not os.path.exists(opt.save):
        os.makedirs(opt.save)
    

    ## forward

    def h(sample):
        inputs = cast(sample[0], opt.dtype).detach()
        targets = cast(sample[1], 'long')
        if opt.teacher_id != '':
            y_s, y_t, distill_losses = f(inputs, targets, params, sample[2])
            loss_groups = {}
            for key in meters_distill.keys():
                loss_groups[key] = [v.sum() for v in distill_losses[key]]
                [m.add(v.item()) for m, v in zip(meters_distill[key], loss_groups[key])]
            return knowledge_distillation(y_s, y_t, targets, opt.temperature, opt.alpha) \
                + opt.beta * sum(loss_groups['AT']) \
                + opt.gamma * sum(loss_groups['SP']) \
                + opt.delta * sum(loss_groups['CSH']) \
                , y_s
        else:
            y = f(inputs, targets, params, sample[2])[0]
            return F.cross_entropy(y, targets), y
    

    ## callbacks in training

    def log(t, state):
        torch.save(dict(params={k: v.data for k, v in params.items()},
                        optimizer=state['optimizer'].state_dict(),
                        epoch=t['epoch']),
                   os.path.join(opt.save, 'model.pt7'))
        z = vars(opt).copy(); z.update(t)
        logname = os.path.join(opt.save, 'log.txt')
        with open(logname, 'a') as f:
            f.write('json_stats: ' + json.dumps(z) + '\n')
        print(z)


    def on_sample(state):
        state['sample'].append(state['train'])


    def on_forward(state):
        classacc.add(state['output'].data, state['sample'][1])
        meter_loss.add(state['loss'].item())


    def on_start(state):
        state['epoch'] = epoch


    def on_start_epoch(state):
        classacc.reset()
        meter_loss.reset()
        timer_train.reset()
        for key in meters_distill.keys():
            [meter.reset() for meter in meters_distill[key]]
        state['iterator'] = tqdm(train_loader)

        epoch = state['epoch'] + 1
        if epoch in epoch_step:
            lr = state['optimizer'].param_groups[0]['lr']
            state['optimizer'] = create_optimizer(opt, lr * opt.lr_decay_ratio)


    def on_end_epoch(state):
        train_loss = meter_loss.mean
        train_acc = classacc.value()
        train_time = timer_train.value()
        meter_loss.reset()
        classacc.reset()
        timer_test.reset()

        engine.test(h, test_loader)

        test_acc = classacc.value()[0]
        print(log({
            "num_classes": num_classes,
            "n_parameters": n_parameters,
            "train_time": train_time,
            "test_time": timer_test.value(),
            "distill_losses": {key:[m.value() for m in meters_distill[key]] for key in meters_distill.keys()},
            "selected_channel":selected[-1],
            "train_loss": train_loss,
            "train_acc": train_acc[0],
            "test_loss": meter_loss.mean,
            "test_acc": test_acc,
            "epoch": state['epoch']
           }, state))
        print('==> id: %s (%d/%d), test_acc: \33[91m%.2f\033[0m' % \
                       (opt.save, state['epoch'], opt.epochs, test_acc))

    ## hook and start training

    engine = Engine()
    engine.hooks['on_sample'] = on_sample
    engine.hooks['on_forward'] = on_forward
    engine.hooks['on_start_epoch'] = on_start_epoch
    engine.hooks['on_end_epoch'] = on_end_epoch
    engine.hooks['on_start'] = on_start
    engine.train(h, train_loader, opt.epochs, optimizer)


if __name__ == '__main__':
    main()