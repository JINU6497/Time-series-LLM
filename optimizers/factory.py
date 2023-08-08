from lion_pytorch import Lion

def create_optimizer(model, opt_name: str, lr: float, params: dict = {}):
    if opt_name in __import__('torch.optim', fromlist='optim').__dict__.keys():
        optimizer = __import__('torch.optim', fromlist='optim').__dict__[opt_name](model.parameters(), lr=lr, **params)
    elif opt_name == 'lion':
        optimizer = Lion(model.parameters(), lr=lr, **params)
    return optimizer        