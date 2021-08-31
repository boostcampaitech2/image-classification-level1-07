import timm 

def get_model(model_name):
    model = timm.create_model(model_name, pretrained = True, num_classes = 18)

    return model

def get_model_list():
    return timm.list_models(pretrained=True)
