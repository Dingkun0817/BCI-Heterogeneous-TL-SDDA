import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def table1_exp():
    data = pd.read_csv('exp_woEA.csv')

    # Extract the dataset names and AUC values
    datasets = data['Approach']
    DAN = data['DAN']
    DANN = data['DANN']
    JAN = data['JAN']
    CDAN = data['CDAN']
    MDD = data['MDD']
    MCC = data['MCC']
    SHOT = data['SHOT']
    ISFDA = data['ISFDA']


    # Standard deviations from the table (Not provided, assuming 0.01 for example)
    Column_spacing = [0.01, 0.01, 0.01, 0.01]

    # Set the width of the bars
    bar_width = 0.1

    # Set the positions of the bars on the x-axis
    num_datasets = len(datasets)
    bar_positions = [np.arange(num_datasets) + i * bar_width for i in range(9)]

    # Create the bar plot
    # colors = ['#F0FFF0', '#B9F9DF', '#79E9DC', '#56DCE4', '#37C8F6', '#25AEFF', '#377EFF', '#0055E6', '#3700D4']
    colors = ['#B9F9DF', '#79E9DC', '#56DCE4', '#37C8F6', '#25AEFF', '#377EFF', '#0055E6', '#3700D4']

    plt.bar(bar_positions[0], DAN, color=colors[0], width=bar_width, edgecolor='grey', yerr=Column_spacing, capsize=4, label='DAN')
    plt.bar(bar_positions[1], DANN, color=colors[1], width=bar_width, edgecolor='grey', yerr=Column_spacing, capsize=4, label='DANN')
    plt.bar(bar_positions[2], JAN, color=colors[2], width=bar_width, edgecolor='grey', yerr=Column_spacing, capsize=4, label='JAN')
    plt.bar(bar_positions[3], CDAN, color=colors[3], width=bar_width, edgecolor='grey', yerr=Column_spacing, capsize=4, label='CDAN')
    plt.bar(bar_positions[4], MDD, color=colors[4], width=bar_width, edgecolor='grey', yerr=Column_spacing, capsize=4, label='MDD')
    plt.bar(bar_positions[5], MCC, color=colors[5], width=bar_width, edgecolor='grey', yerr=Column_spacing, capsize=4, label='MCC')
    plt.bar(bar_positions[6], SHOT, color=colors[6], width=bar_width, edgecolor='grey', yerr=Column_spacing, capsize=4, label='SHOT')
    plt.bar(bar_positions[7], ISFDA, color=colors[7], width=bar_width, edgecolor='grey', yerr=Column_spacing, capsize=4, label='ISFDA')
    # plt.bar(bar_positions[8], ours_acc, color=colors[8], width=bar_width, edgecolor='grey', yerr=Column_spacing, capsize=4, label='UMMAN(Ours)')

    # Add labels, title, and legend
    plt.xlabel('Approaches')
    plt.ylabel('Percentage')
    plt.title('Comparison with Various Sophisticated Approaches without EA')
    plt.xticks([r + 3 * bar_width for r in range(len(datasets))], datasets)

    plt.ylim(40, 90)    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    plt.yticks(np.arange(40, 90, step=10))
    plt.legend()

    plt.axhline(y=MCC[0]+0.01, color='red', linestyle='dashed', xmin=0.04, xmax=0.24)
    plt.axhline(y=MCC[1]+0.01, color='red', linestyle='dashed', xmin=0.28, xmax=0.48)
    plt.axhline(y=MDD[2]+0.01, color='red', linestyle='dashed', xmin=0.52, xmax=0.716)
    plt.axhline(y=ISFDA[3]+0.01, color='red', linestyle='dashed', xmin=0.763, xmax=0.957)
    # plt.axhline(y=rf_acc[4]+0.01, color='red', linestyle='dashed', xmin=0.8, xmax=0.953)

    # Show the plot
    plt.tight_layout()
    # plt.show()

def table2_exp():

    # plt.figure(dpi=600)

    data = pd.read_csv('exp_subject_EA.csv')

    # Extract the dataset names and AUC values
    datasets = data['Approach']
    DAN = data['DAN']
    DANN = data['DANN']
    JAN = data['JAN']
    CDAN = data['CDAN']
    MDD = data['MDD']
    MCC = data['MCC']
    SHOT = data['SHOT']
    ISFDA = data['ISFDA']
    Ours = data['Ours']


    # Standard deviations from the table (Not provided, assuming 0.01 for example)
    Column_spacing = [0.01, 0.01, 0.01, 0.01]

    # Set the width of the bars
    bar_width = 0.1

    # Set the positions of the bars on the x-axis
    num_datasets = len(datasets)
    bar_positions = [np.arange(num_datasets) + i * bar_width for i in range(9)]

    # Create the bar plot
    colors = ['#F0FFF0', '#B9F9DF', '#79E9DC', '#56DCE4', '#37C8F6', '#25AEFF', '#377EFF', '#0055E6', '#3700D4']
    # colors = ['#B9F9DF', '#79E9DC', '#56DCE4', '#37C8F6', '#25AEFF', '#377EFF', '#0055E6', '#3700D4']

    plt.bar(bar_positions[0], DAN, color=colors[0], width=bar_width, edgecolor='grey', yerr=Column_spacing, capsize=4, label='DAN')
    plt.bar(bar_positions[1], DANN, color=colors[1], width=bar_width, edgecolor='grey', yerr=Column_spacing, capsize=4, label='DANN')
    plt.bar(bar_positions[2], JAN, color=colors[2], width=bar_width, edgecolor='grey', yerr=Column_spacing, capsize=4, label='JAN')
    plt.bar(bar_positions[3], CDAN, color=colors[3], width=bar_width, edgecolor='grey', yerr=Column_spacing, capsize=4, label='CDAN')
    plt.bar(bar_positions[4], MDD, color=colors[4], width=bar_width, edgecolor='grey', yerr=Column_spacing, capsize=4, label='MDD')
    plt.bar(bar_positions[5], MCC, color=colors[5], width=bar_width, edgecolor='grey', yerr=Column_spacing, capsize=4, label='MCC')
    plt.bar(bar_positions[6], SHOT, color=colors[6], width=bar_width, edgecolor='grey', yerr=Column_spacing, capsize=4, label='SHOT')
    plt.bar(bar_positions[7], ISFDA, color=colors[7], width=bar_width, edgecolor='grey', yerr=Column_spacing, capsize=4, label='ISFDA')
    plt.bar(bar_positions[8], Ours, color=colors[8], width=bar_width, edgecolor='grey', yerr=Column_spacing, capsize=4, label='Ours')

    # Add labels, title, and legend
    plt.xlabel('Approaches')
    plt.ylabel('Percentage')
    plt.title('Comparison with Various Sophisticated Approaches with EA')
    plt.xticks([r + 3 * bar_width for r in range(len(datasets))], datasets)

    plt.ylim(60, 75)    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    plt.yticks(np.arange(60, 76, step=5))
    plt.legend()

    plt.axhline(y=SHOT[0]+0.01, color='red', linestyle='dashed', xmin=0.04, xmax=0.24)
    plt.axhline(y=MCC[1]+0.01, color='red', linestyle='dashed', xmin=0.28, xmax=0.48)
    plt.axhline(y=SHOT[2]+0.01, color='red', linestyle='dashed', xmin=0.52, xmax=0.716)
    plt.axhline(y=SHOT[3]+0.01, color='red', linestyle='dashed', xmin=0.763, xmax=0.957)
    # plt.axhline(y=rf_acc[4]+0.01, color='red', linestyle='dashed', xmin=0.8, xmax=0.953)

    # Show the plot
    plt.tight_layout()
    # plt.savefig('ours_ea.png')
    # plt.show()


# def table1():
#
#     # Create a 1x2 subplot grid
#     fig, axes = plt.subplots(1, 2, figsize=(16, 6))
#     # plt.figure(figsize=(16, 6))
#     plt.subplot(1, 2, 1)
#     table1_exp()
#
#     plt.subplot(1, 2, 2)
#     table2_exp()
#
#     plt.tight_layout()
#     # plt.savefig('res.png')
#     plt.show()

def table1():
    plt.figure(figsize=(16, 6))

    plt.subplot(1, 2, 1)
    table1_exp()

    plt.subplot(1, 2, 2)
    table2_exp()
    plt.savefig('exp.png')
    plt.show()

if __name__ == '__main__':
    table1()   # wo_ea和with_ea画一起
    # table2_exp()  ## with_ea
    pass