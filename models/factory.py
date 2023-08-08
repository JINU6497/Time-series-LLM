def create_model(modelname: str, params: dict = {}):
    model = __import__('models').__dict__[modelname](params)
    return model