def create_criterion(loss_name: str, params: dict = {}):
    if loss_name in __import__('torch.nn', fromlist='nn').__dict__.keys():
        criterion = __import__('torch.nn', fromlist='nn').__dict__[loss_name](**params)

    return criterion