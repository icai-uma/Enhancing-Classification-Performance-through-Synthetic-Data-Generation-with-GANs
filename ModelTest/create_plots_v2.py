import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
sns.set_theme(style="darkgrid", font_scale=2)
import pandas as pd


nets = ['og', 'aug', 'began', 'progan', 'regan_progan']
fig, axes = plt.subplots(len(nets), 1)
for tech, ax in zip(nets, axes):
    inception_data = pd.read_csv(f'figures_{tech}/inception/train_metrics.csv')
    resnet_data = pd.read_csv(f'figures_{tech}/resnet/train_metrics.csv')
    vgg16_data = pd.read_csv(f'figures_{tech}/vgg16/train_metrics.csv')

    inception_data['Model'] = ['Inception'] * len(inception_data)
    resnet_data['Model'] = ['ResNet'] * len(resnet_data)
    vgg16_data['Model'] = ['VGG16'] * len(vgg16_data)

    data = pd.concat([inception_data, resnet_data, vgg16_data])

    # plt.figure(figsize=(20, 11.25))
    if tech == 'og':
        ax1 = sns.lineplot(x='Epoch', y='AccVal', hue='Model', data=data, palette='colorblind', ax=ax)
        sns.move_legend(ax1, "upper left", bbox_to_anchor=(1, 1))
    else:
        sns.lineplot(x='Epoch', y='AccVal', hue='Model', data=data, palette='colorblind', ax=ax, legend=False)

# plt.savefig('fig_all.png')
plt.show()

