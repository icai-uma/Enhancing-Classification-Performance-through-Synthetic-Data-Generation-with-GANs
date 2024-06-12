import pandas as pd
import os


tech = 'progan2'
os.makedirs(f'cf_metrics/{tech}', exist_ok=True)

inception_data = pd.read_csv(f'figures_{tech}/inception/cf_mat.csv')
resnet_data = pd.read_csv(f'figures_{tech}/resnet/cf_mat.csv')
vgg16_data = pd.read_csv(f'figures_{tech}/vgg16/cf_mat.csv')

for name, cf in zip(['inception', 'resnet', 'vgg16'], [inception_data, resnet_data, vgg16_data]):
    # Transform dataframe to numpy array
    cf = cf.to_numpy()
    # Shape will be always 2x2
    # Compute precision, recall, f1-score
    TP = cf[0][0]
    FP = cf[0][1]
    FN = cf[1][0]
    TN = cf[1][1]
    
    print(f'TP: {TP}, FP: {FP}, FN: {FN}, TN: {TN}')
    
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * (precision * recall) / (precision + recall)
    # Save metrics to pandas dataframe
    df = pd.DataFrame({'Precision': [precision], 'Recall': [recall], 'F1-Score': [f1_score]})
    df.to_csv(f'cf_metrics/{tech}/cf_metrics_{name}.csv', index=False)
    
