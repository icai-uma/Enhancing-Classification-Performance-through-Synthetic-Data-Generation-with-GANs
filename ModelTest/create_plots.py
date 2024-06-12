import seaborn as sns
import pandas as pd
from matplotlib import pyplot as plt
sns.set_theme(style="darkgrid", font_scale=2.5)
import pandas as pd

tech = 'down'
inception_data_og = pd.read_csv('figures_og/inception/train_metrics.csv')
resnet_data_og = pd.read_csv('figures_og/resnet/train_metrics.csv')
vgg16_data_og = pd.read_csv('figures_og/vgg16/train_metrics.csv')

inception_data = pd.read_csv(f'figures_{tech}/inception/train_metrics.csv')
resnet_data = pd.read_csv(f'figures_{tech}/resnet/train_metrics.csv')
vgg16_data = pd.read_csv(f'figures_{tech}/vgg16/train_metrics.csv')

inception_data_og['Model'] = ['Inception'] * len(inception_data)
resnet_data_og['Model'] = ['ResNet'] * len(resnet_data)
vgg16_data_og['Model'] = ['VGG16'] * len(vgg16_data)

inception_data['Model'] = ['Inception'] * len(inception_data)
resnet_data['Model'] = ['ResNet'] * len(resnet_data)
vgg16_data['Model'] = ['VGG16'] * len(vgg16_data)


data_og = pd.concat([inception_data_og, resnet_data_og, vgg16_data_og])
data = pd.concat([inception_data, resnet_data, vgg16_data])
# df.to_csv('began_train_data.csv', index=False)


# data = pd.read_csv('figures_began/inception/train_metrics.csv')
# data = pd.read_csv('began_train_data.csv')

# plt.figure(figsize=(20, 11.25))
fig, ax = plt.subplots(2, 1)
fig.set_size_inches(27, 11.25)
ax1 = sns.lineplot(x='Epoch', y='AccVal', hue='Model', data=data_og, palette='colorblind', ax=ax[0])
sns.lineplot(x='Epoch', y='AccVal', hue='Model', data=data, palette='colorblind', ax=ax[1], legend=False)
sns.move_legend(ax1, "lower center", bbox_to_anchor=(.5, 1), ncol=3, title=None, frameon=False)
# plt.savefig(f'fig_{tech}.png')
plt.show()

