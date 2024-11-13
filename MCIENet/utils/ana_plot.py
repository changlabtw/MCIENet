import pandas as pd
import seaborn as sns 

import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from matplotlib.pyplot import MultipleLocator

def plot_train_val_loss(train_total_loss, val_total_loss, file_path):
    plt.clf()
    plt.figure(figsize=(10, 5))

    plt.plot(range(1, len(train_total_loss) + 1), train_total_loss, c='b', marker='s', label='Train')
    plt.plot(range(1, len(val_total_loss) + 1), val_total_loss, c='r', marker='o', label='Validation')
    plt.legend(loc='best')
    plt.title('Train loss vs Validation loss')

    # gca() 獲取當前的 axis
    plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))  # 不要小數點
    plt.gca().xaxis.set_major_locator(MultipleLocator(1)) # 間隔1


    plt.savefig(file_path)

def plot_confusion_matrix(y_actual:list, y_predicted:list, file_path:str, class_ls:list=None):
    data = {'y_actual': y_actual,
            'y_predicted': y_predicted}

    df = pd.DataFrame(data)
    df_cm = pd.crosstab(df['y_actual'], df['y_predicted'], rownames=['Actual'], colnames=['Predicted'])

    if class_ls:
        df_cm = pd.DataFrame(df_cm, index=class_ls, columns=class_ls)

    plt.figure(figsize = (10,10))  #This is the size of the image
    heatM = sns.heatmap(df_cm, vmin = -1, vmax = 1,center = 0, cmap = sns.diverging_palette(20, 220, n = 200),  square = True, annot = True) #this are the caracteristics of the heatmap
    # heatM = sns.heatmap(df_cm, vmin = -1, vmax = 1,center = 0, cmap = 'PuRd',  square = True, annot = True) #this are the caracteristics of the heatmap

    # heatM.set_ylim([10,0]) # This is the limit in y axis (number of features)

    plt.savefig(file_path)
