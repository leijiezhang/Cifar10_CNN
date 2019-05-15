import torch.optim as optim


lr_lambda = lambda epoch: 0.95 ** epoch


def _get_optimizer(net, lr, opt_type, is_lr_adjust = False, lr_adjust_mtd = 'Cos', lr_lambda=lr_lambda):
    Optimizers = {
        'SGD': optim.SGD(net.parameters(), lr=lr, momentum=0.9),
        'Adam': optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.99)),
        'LBFGS': optim.LBFGS(net.parameters(), lr=lr),
        'RMSprop': optim.RMSprop(net.parameters(), lr=lr),
        'Rprop': optim.Rprop(net.parameters(), lr=lr),
        'Adagrad': optim.Adagrad(net.parameters(), lr=lr),
    }
    optimizer = Optimizers[opt_type]

    if(is_lr_adjust):
        lr_schedulers = {
            'LambdaLR': optim.lr_scheduler.LambdaLR(optimizer, lr_lambda),
            'StepLR': optim.lr_scheduler.StepLR(optimizer, 3),
            'MultiStepLR': optim.lr_scheduler.MultiStepLR(optimizer, [10, 30]),
            'ExponentialLR': optim.lr_scheduler.ExponentialLR(optimizer, 0.1),
            'Cos': optim.lr_scheduler.CosineAnnealingLR(optimizer, 200),
            # 'CyclicLR': optim.lr_scheduler.CyclicLR(optimizer, 0.001, 0.01),
        }
        scheduler = lr_schedulers[lr_adjust_mtd]
    else:
        scheduler = None

    return scheduler, optimizer
