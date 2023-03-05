# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 11:38:30 2023

@author: fcolo
"""

## VISUALIZE CLUSTERS

# Make the required imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap,LinearSegmentedColormap

# Load the file csv with cluster labels previosly created and convert from a DataFrame to a numpy array
data_plot = np.array(pd.read_csv('path/to/database/with/clusters/labels', sep=','))

# Create new hsv colormaps in range of 0.3 (green) to 0.7 (blue)
hsv_modified = cm.get_cmap('hsv', 256)
newcmp = ListedColormap(hsv_modified(np.linspace(0.1, 0.63, 256)))

# Create a scatter plot of the first two feature of the database. Points are coloured based on clusters' membership
scatter = plt.scatter(data_plot[:,3], data_plot[:,4], c=data_plot[:, -1], cmap=newcmp)
plt.legend(*scatter.legend_elements())
plt.title('Clustering labels')
plt.show()
