import numpy as np
import matplotlib.pyplot as plt

def smooth(data, window=5):
    """
    Smooth input array or list and return same type.
    """
    is_array = False
    if isinstance(data, np.ndarray):
        is_array = True
    else:
        data = np.array(data)

    data_smooth = []
    for i in range(len(data)):
        indices = list(range(max(i - window + 1, 0),
                             min(i + window + 1, len(data))))
        avg = 0
        for j in indices:
            avg += data[j]
        avg /= float(len(indices))
        data_smooth.append(avg)
    if is_array:
        return np.array(data_smooth)
    else:
        return data_smooth

def set_default_mpl():
    from matplotlib import cycler
    colors = cycler('color',
                    ['#EE6666', '#3388BB', '#9988DD',
                     '#EECC55', '#88BB44', '#FFBBBB'])
    plt.rc('axes', facecolor='#E6E6E6', edgecolor='none',
           axisbelow=True, grid=True, prop_cycle=colors)
    plt.rc('grid', color='w', linestyle='solid')
    plt.rc('xtick', direction='out', color='gray')
    plt.rc('ytick', direction='out', color='gray')
    plt.rc('patch', edgecolor='#E6E6E6')

    plt.rcParams.update({'font.size': 24})
    plt.rcParams.update({'xtick.labelsize': 20})
    plt.rcParams.update({'ytick.labelsize': 20})
    plt.rcParams.update({'axes.titlesize': 24})
    plt.rcParams.update({'axes.labelsize': 20})
    plt.rcParams.update({'lines.linewidth': 5})