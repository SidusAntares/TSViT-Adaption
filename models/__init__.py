from models.TSViT.TSViTdense import TSViT, TSViT_mlp_da
from models.TSViT.TSViTdense_extract import TSViT_extract


def get_model(config, device):
    model_config = config['MODEL']
    if model_config['architecture'] == "TSViT":
        return TSViT(model_config).to(device)
    elif model_config['architecture'] == "TSViT_mlp_da":
        return TSViT_mlp_da(model_config).to(device)
    else:
        raise NameError("Model architecture %s not found, choose from: 'TSViT'")
