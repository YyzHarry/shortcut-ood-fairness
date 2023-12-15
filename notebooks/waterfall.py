"""
Modified from https://github.com/chrispaulca/waterfall
"""
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot(index, data, std=None, title="", x_lab="", y_lab="", formatting="{:,.1f}",
         id_color='#94A7CF', ood_color='#8C82BC', green_color='#6AAE74', red_color='#CE2417', net_label='net',
         rotation_value=30, blank_color=(0, 0, 0, 0), figsize=(10, 10), fontsize=10, ticksize=20,
         draw_intermediate_lines=True,
         arrow_idxs=[],  # list of strings in index to be replaced by an arrow (instead of a bar)
         arrow_args={}, ebar_args={}):

    index = np.array(index)
    data = np.array(data)

    changes = {'amount': data}

    # define format formatter
    def fmt(x, pos):
        """The two args are the value and tick position"""
        return formatting.format(x)
    formatter = FuncFormatter(fmt)

    fig, ax = plt.subplots(figsize=figsize)
    ax.yaxis.set_major_formatter(formatter)

    # Store data and create a blank series to use for the waterfall
    trans = pd.DataFrame(data=changes, index=index)
    blank = trans.amount.cumsum().shift(1).fillna(0)
    
    trans['positive'] = trans['amount'] > 0

    # Get the net total number for the final element in the waterfall
    total = trans.sum().amount
    trans.loc[net_label] = total
    blank.loc[net_label] = total

    # The steps graphically show the levels as well as used for label placement
    step = blank.reset_index(drop=True).repeat(3).shift(-1)
    step[1::3] = np.nan

    # When plotting the last element, we want to show the full bar, set the blank to 0
    blank.loc[net_label] = 0
    
    # define bar colors for net bar
    trans.loc[trans['positive'] > 1, 'positive'] = 99
    trans.loc[trans['positive'] < 0, 'positive'] = 99
    trans.loc[(trans['positive'] > 0) & (trans['positive'] < 1), 'positive'] = 99

    trans['color'] = trans['positive']
    trans.loc[trans['positive'] == 1, 'color'] = green_color
    trans.loc[trans['positive'] == 0, 'color'] = red_color
    trans.loc[trans['positive'] == 99, 'color'] = ood_color
    trans['arrow'] = trans.index.isin(arrow_idxs)

    my_colors = list(trans.color)
    my_colors[0] = id_color

    # Plot and label    
    plt.bar(range(0, len(trans.index)), blank, width=0.5, color=blank_color)
    plt.bar(range(0, len(trans.index)), trans.amount * (1 - trans.arrow), width=0.6, bottom=blank, color=my_colors)
    
    # draw arrows
    for c, (idx, row) in enumerate(trans.iterrows()):
        if row['arrow']:
            plt.arrow(c, blank[idx], 0, row['amount'], color=row['color'], length_includes_head=True,
                      head_length=(ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.3 / 6, **arrow_args)
    
    if std is not None:
        # do not plot stds for arrows
        std[1] = 0
        std[2] = 0
        plt.errorbar(range(0, len(trans.index)), blank + trans.amount, yerr=std, **ebar_args)

    # axis labels
    plt.xlabel("\n" + x_lab)
    plt.ylabel(y_lab + "\n")

    # Get the y-axis position for the labels
    y_height = trans.amount.cumsum().shift(1).fillna(0)

    temp = list(trans.amount)

    # create dynamic chart range
    for i in range(len(temp)):
        if (i > 0) & (i < (len(temp) - 1)):
            temp[i] = temp[i] + temp[i-1]

    trans['temp'] = temp
     
    if std is not None:
        plot_max = (trans['temp'] + std).max()
        plot_min = (trans['temp'] - std).min()
    else:
        plot_max = (trans['temp']).max()
        plot_min = (trans['temp']).min()
    
    # Make sure the plot doesn't accidentally focus only on the changes in the data
    if all(i >= 0 for i in temp):
        plot_min = 0
    if all(i < 0 for i in temp):
        plot_max = 0

    if abs(plot_max) >= abs(plot_min):
        maxmax = abs(plot_max)
    else:
        maxmax = abs(plot_min)

    pos_offset = maxmax / 40
    plot_offset = maxmax / 15  # needs to me cumulative sum dynamic

    # Start label loop
    loop = 0
    for i, (index, row) in enumerate(trans.iterrows()):
        # For the last item in the list, we don't want to double count
        if row['amount'] == total:
            y = y_height[loop]
        else:
            y = y_height[loop] + row['amount']
        # Determine if we want a neg or pos offset
        if row['amount'] > 0:
            y += (pos_offset * 2 + std[i])
            plt.annotate(formatting.format(row['amount']), (loop, y), ha="center",
                         color=green_color if 0 < i < 3 else ood_color, fontsize=fontsize)
        else:
            y -= (pos_offset * 4 + std[i])
            plt.annotate(formatting.format(row['amount']), (loop, y), ha="center",
                         color=red_color if 0 < i < 3 else id_color, fontsize=fontsize)
        loop += 1

    # Scale up the y axis so there is room for the labels
    plt.xlim(ax.get_xlim()[0] - .3, ax.get_xlim()[1] + .3)
    plt.ylim(plot_min - round(3.6*plot_offset, 7), plot_max + round(3.6*plot_offset, 7))

    # Rotate the labels
    plt.xticks(range(0, len(trans)), trans.index, rotation=rotation_value)

    # set tick size
    ax.tick_params('x', labelsize=ticksize)
    ax.tick_params('y', labelsize=ticksize)

    # only display ID / OOD xlabel
    first_label = trans.index[0]
    last_label = trans.index[-1]
    ax.set_xticks([0, len(trans) - 1])
    ax.set_xticklabels([first_label, last_label])

    # remove top and right spines
    ax.spines[['right', 'top']].set_visible(False)
    ax.set_axisbelow(True)

    # add zero line and title
    plt.axhline(0, color='gray', linewidth=1, linestyle='-', alpha=0.4)

    # draw intermediate dashed lines
    if draw_intermediate_lines:
        for c, i in enumerate(trans.amount.cumsum().values):
            if c != trans.shape[0] - 1:
                plt.hlines(i, xmin=c, xmax=c+1, color='black', linewidth=1.5, linestyle="dashed")

    plt.title(title)
    plt.tight_layout()
    return fig, ax
