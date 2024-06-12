import pandas as pd



inception_data = pd.read_csv('figures_down/inception/test_metrics.csv')
resnet_data = pd.read_csv('figures_down/resnet/test_metrics.csv')
vgg16_data = pd.read_csv('figures_down/vgg16/test_metrics.csv')

inception_data['Model'] = ['Inception'] * len(inception_data)
resnet_data['Model'] = ['ResNet'] * len(resnet_data)
vgg16_data['Model'] = ['VGG16'] * len(vgg16_data)


data = pd.concat([inception_data, resnet_data, vgg16_data])


data = data.groupby(['Model']).agg(['mean', 'std']).reset_index()

data.to_csv('down.csv')
