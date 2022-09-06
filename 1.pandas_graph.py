import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import normal,rand

x = np.arange(3)
years = ['2017', '2018', '2019']
values = [100, 400, 900]

plt.bar(x, values, width=0.5, align='edge', color="springgreen",
        edgecolor="gray", linewidth=10, tick_label=years)
plt.show()
