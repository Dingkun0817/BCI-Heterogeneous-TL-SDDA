import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def table1_AUC():
    data = pd.read_csv('model_aucs.csv')

    # Extract the dataset names and AUC values
    datasets = data['Dataset']
    rf_aucs = data['RF']
    svm_aucs = data['SVM']
    fcnn_aucs = data['FC-NN']
    fcnn_svm_aucs = data['FC-NN+SVM']
    cnn_aucs = data['CNN']
    cnn_rf_aucs = data['CNN+RF']
    TaxoNN_aucs = data['TaxoNN']
    graphsage_aucs = data['GraphSAGE']
    ours_aucs = data['UMMAN(Ours)']

    # Standard deviations from the table (Not provided, assuming 0.01 for example)
    clom = np.array([0.01, 0.01, 0.01, 0.01, 0.01])

    # Set the width of the bars
    bar_width = 0.1

    # Set the positions of the bars on the x-axis
    num_datasets = len(datasets)
    bar_positions = [np.arange(num_datasets) + i * bar_width for i in range(9)]

    # Create the bar plot
    colors = ['#F0FFF0', '#B9F9DF', '#79E9DC', '#56DCE4', '#37C8F6', '#25AEFF', '#377EFF', '#0055E6', '#3700D4']

    plt.bar(bar_positions[0], rf_aucs, color=colors[0], width=bar_width, edgecolor='grey', yerr=clom, capsize=4, label='RF')
    plt.bar(bar_positions[1], svm_aucs, color=colors[1], width=bar_width, edgecolor='grey', yerr=clom, capsize=4, label='SVM')
    plt.bar(bar_positions[2], fcnn_aucs, color=colors[2], width=bar_width, edgecolor='grey', yerr=clom, capsize=4, label='FC-NN')
    plt.bar(bar_positions[3], fcnn_svm_aucs, color=colors[3], width=bar_width, edgecolor='grey', yerr=clom, capsize=4, label='FC-NN+SVM')
    plt.bar(bar_positions[4], cnn_aucs, color=colors[4], width=bar_width, edgecolor='grey', yerr=clom, capsize=4, label='CNN')
    plt.bar(bar_positions[5], cnn_rf_aucs, color=colors[5], width=bar_width, edgecolor='grey', yerr=clom, capsize=4, label='CNN+RF')
    plt.bar(bar_positions[6], TaxoNN_aucs, color=colors[6], width=bar_width, edgecolor='grey', yerr=clom, capsize=4, label='TaxoNN')
    plt.bar(bar_positions[7], graphsage_aucs, color=colors[7], width=bar_width, edgecolor='grey', yerr=clom, capsize=4, label='GraphSAGE')
    plt.bar(bar_positions[8], ours_aucs, color=colors[8], width=bar_width, edgecolor='grey', yerr=clom, capsize=4, label='UMMAN(Ours)')

    # Add labels, title, and legend
    plt.xlabel('Datasets')
    plt.ylabel('AUC')
    plt.title('AUC with Error Bars for Different Models and Datasets')
    plt.xticks([r + 4 * bar_width for r in range(len(datasets))], datasets)
    plt.legend()

    plt.axhline(y=rf_aucs[0]+0.01, color='red', linestyle='dashed', xmin=0.04, xmax=0.21)
    plt.axhline(y=rf_aucs[1]+0.01, color='red', linestyle='dashed', xmin=0.23, xmax=0.4)
    plt.axhline(y=rf_aucs[2]+0.01, color='red', linestyle='dashed', xmin=0.42, xmax=0.585)
    plt.axhline(y=rf_aucs[3]+0.01, color='red', linestyle='dashed', xmin=0.6, xmax=0.77)
    plt.axhline(y=rf_aucs[4]+0.01, color='red', linestyle='dashed', xmin=0.8, xmax=0.953)

    # Show the plot
    plt.tight_layout()

def table1_Accuracy():
    data = pd.read_csv('model_aucs.csv')

    # Extract the dataset names and AUC values
    datasets = data['Dataset']
    rf_acc = data['RF']
    svm_acc = data['SVM']
    fcnn_acc = data['FC-NN']
    fcnn_svm_acc = data['FC-NN+SVM']
    cnn_acc = data['CNN']
    cnn_rf_acc = data['CNN+RF']
    TaxoNN_acc = data['TaxoNN']
    graphsage_acc = data['GraphSAGE']
    ours_acc = data['UMMAN(Ours)']

    # Standard deviations from the table (Not provided, assuming 0.01 for example)
    Column_spacing = [0.01, 0.01, 0.01, 0.01, 0.01]

    # Set the width of the bars
    bar_width = 0.1

    # Set the positions of the bars on the x-axis
    num_datasets = len(datasets)
    bar_positions = [np.arange(num_datasets) + i * bar_width for i in range(9)]

    # Create the bar plot
    colors = ['#F0FFF0', '#B9F9DF', '#79E9DC', '#56DCE4', '#37C8F6', '#25AEFF', '#377EFF', '#0055E6', '#3700D4']

    plt.bar(bar_positions[0], rf_acc, color=colors[0], width=bar_width, edgecolor='grey', yerr=Column_spacing, capsize=4, label='RF')
    plt.bar(bar_positions[1], svm_acc, color=colors[1], width=bar_width, edgecolor='grey', yerr=Column_spacing, capsize=4, label='SVM')
    plt.bar(bar_positions[2], fcnn_acc, color=colors[2], width=bar_width, edgecolor='grey', yerr=Column_spacing, capsize=4, label='FC-NN')
    plt.bar(bar_positions[3], fcnn_svm_acc, color=colors[3], width=bar_width, edgecolor='grey', yerr=Column_spacing, capsize=4, label='FC-NN+SVM')
    plt.bar(bar_positions[4], cnn_acc, color=colors[4], width=bar_width, edgecolor='grey', yerr=Column_spacing, capsize=4, label='CNN')
    plt.bar(bar_positions[5], cnn_rf_acc, color=colors[5], width=bar_width, edgecolor='grey', yerr=Column_spacing, capsize=4, label='CNN+RF')
    plt.bar(bar_positions[6], TaxoNN_acc, color=colors[6], width=bar_width, edgecolor='grey', yerr=Column_spacing, capsize=4, label='TaxoNN')
    plt.bar(bar_positions[7], graphsage_acc, color=colors[7], width=bar_width, edgecolor='grey', yerr=Column_spacing, capsize=4, label='GraphSAGE')
    plt.bar(bar_positions[8], ours_acc, color=colors[8], width=bar_width, edgecolor='grey', yerr=Column_spacing, capsize=4, label='UMMAN(Ours)')

    # Add labels, title, and legend
    plt.xlabel('Datasets')
    plt.ylabel('Accuracy')
    plt.title('Accuracy with Error Bars for Different Models and Datasets')
    plt.xticks([r + 4 * bar_width for r in range(len(datasets))], datasets)
    plt.legend()

    plt.axhline(y=rf_acc[0]+0.01, color='red', linestyle='dashed', xmin=0.04, xmax=0.21)
    plt.axhline(y=fcnn_svm_acc[1]+0.01, color='red', linestyle='dashed', xmin=0.23, xmax=0.4)
    plt.axhline(y=fcnn_acc[2]+0.01, color='red', linestyle='dashed', xmin=0.42, xmax=0.585)
    plt.axhline(y=rf_acc[3]+0.01, color='red', linestyle='dashed', xmin=0.6, xmax=0.77)
    plt.axhline(y=rf_acc[4]+0.01, color='red', linestyle='dashed', xmin=0.8, xmax=0.953)

    # Show the plot
    plt.tight_layout()
    plt.show()

def table1():

    # Create a 1x2 subplot grid
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Call the table1_AUC function in the first subplot
    plt.subplot(1, 2, 1)
    table1_AUC()

    # Call the table1_Accuracy function in the second subplot
    plt.subplot(1, 2, 2)
    table1_Accuracy()

    # plt.savefig('table1_last.pdf', format='pdf', dpi=300)
    plt.tight_layout()
    plt.show()


def table3_plots(data_files, titles):
    num_plots = len(data_files)

    # Set the Seaborn theme globally
    sns.set_theme(style="whitegrid")

    # Set a custom color palette
    custom_palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    # Create a custom gradient background
    colors_bg = sns.color_palette("Blues")
    gradient = LinearSegmentedColormap.from_list("custom_gradient", colors_bg)
    cmap = plt.cm.colors.LinearSegmentedColormap.from_list("custom_gradient", colors_bg)


    # Create subplots
    fig, axes = plt.subplots(1, num_plots, figsize=(25,4 ))

    for i in range(num_plots):
        df = pd.read_csv(data_files[i])
        colors = sns.color_palette("husl", n_colors=len(df['Method']))

        # Create a KDE plot
        sns.set(style="whitegrid")
        sns.kdeplot(data=df, x='Accuracy', y='AUC', fill=True, cmap=cmap, thresh=0.001, ax=axes[i])

        # Add individual data points with distinct colors and labels
        for j, row in df.iterrows():
            method = row['Method']
            x = row['Accuracy']
            y = row['AUC']
            axes[i].scatter(x=[x], y=[y], color=colors[j], label=method, s=80)
            if j == 0:
                axes[i].annotate(method, (x, y), textcoords="offset points", xytext=(95, -8), ha='right', fontsize=16,
                                 color=colors[j])
            if j == 1:
                axes[i].annotate(method, (x, y), textcoords="offset points", xytext=(-8, -4), ha='right', fontsize=16,
                                 color=colors[j])
            if j == 2:
                axes[i].annotate(method, (x, y), textcoords="offset points", xytext=(-10, -14), ha='right', fontsize=16,
                                 color=colors[j])
            if j == 3:
                axes[i].annotate(method, (x, y), textcoords="offset points", xytext=(85, -0), ha='right', fontsize=16,
                                 color=colors[j])

        # Set plot attributes
        axes[i].set_title(titles[i], fontsize=10)
        axes[i].legend(loc='upper left', fontsize=10)

    # plt.savefig('table3_last.pdf', format='pdf', dpi=300)
    plt.tight_layout()
    plt.show()



def table3():
    data_files = ['table3_Cirrhosis.csv', 'table3_IBD.csv', 'table3_Obesity.csv', 'table3_T2D.csv', 'table3_WT2D.csv']
    titles = ['Ablation Study on Attention Block and NFGI Module-Cirrhosis',
              'Ablation Study on Attention Block and NFGI Module-IBD',
              'Ablation Study on Attention Block and NFGI Module-Obesity',
              'Ablation Study on Attention Block and NFGI Module-T2D',
              'Ablation Study on Attention Block and NFGI Module-WT2D']

    # Call the function to create combined plots
    table3_plots(data_files, titles)

if __name__ == '__main__':
    table1()
    # table3()
    # table1_Accuracy()
    pass




