import pandas as pd
import matplotlib.pyplot as plt

germany_path = '/data/user/ViT/D3/test/ndvi/Germany_ndvi_doy.csv'
pastis_path = '/data/user/ViT/D3/test/ndvi/PASTIS_ndvi_doy.csv'

germany = pd.read_csv(germany_path, sep=',')
pastis = pd.read_csv(pastis_path, sep=',')
doy = pd.Series(germany.columns)[1:]

for i in range(0,6):
    crop_ger = germany.iloc[i,0]
    crop_pas = pastis.iloc[i,0]
    ger = germany.iloc[i,1:].to_numpy()
    pas = pastis.iloc[i,1:].to_numpy()
    mask_ger = (ger > -1) & (ger < 1)
    mask_pas = (pas > -1) & (pas < 1)
    ger = ger[mask_ger]
    doy_ger = doy[mask_ger]
    pas = pas[mask_pas]
    doy_pas = doy[mask_pas]
    # print("doy长度:", len(doy))
    # print("ger长度:", len(ger))
    # print(type(ger)) # <class 'pandas.core.series.Series'>
    plt.figure(figsize=(10, 5))  # 设置图像大小（可选）
    # plt.plot(doy_ger, ger, marker='o', linestyle='-', color='b')
    plt.plot(doy_ger, ger)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    break

