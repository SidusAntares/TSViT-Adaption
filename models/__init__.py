from models.TSViT.TSViTdense import TSViT
from models.TSViT.TSViTdense_extract import TSViT_extract


def get_model(config, device):
    model_config = config['MODEL']
    if model_config['architecture'] == "TSViT":
        return TSViT(model_config).to(device)

    else:
        raise NameError("Model architecture %s not found, choose from: 'TSViT'")
