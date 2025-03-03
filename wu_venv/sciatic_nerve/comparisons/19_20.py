from SDS00019 import plot_subplot1  
from SDS00020 import plot_subplot2

import matplotlib.pyplot as plt

file_path_1 = "wu_venv/csv_files/SDS00019.csv"  
file_path_2 = "wu_venv/csv_files/SDS00020.csv"  

fig, axs = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(10, 8))  

plot_subplot1(axs[0], file_path_1)  
plot_subplot2(axs[1], file_path_2)  

plt.tight_layout()

for ax in axs:
    ax.spines['top'].set_visible(False)     
    ax.spines['right'].set_visible(False)   
    ax.spines['left'].set_visible(False)    
    ax.spines['bottom'].set_visible(False) 
    

def add_scale_bar(ax, x_length, y_length, x_label, y_label):
    x_start = 0.005  # near origin
    y_start = -0.065  # below data 
    
    # horizontal line for scale bar
    ax.plot([x_start, x_start + x_length], [y_start, y_start], color='black', lw=1.5)
    ax.text(x_start + x_length / 2, y_start - 0.005, x_label, 
            ha='center', va='top', fontsize=10)  
    
    # vertical line for scale bar
    ax.plot([x_start, x_start], [y_start, y_start + y_length], color='black', lw=1.5)
    ax.text(x_start - 0.0005, y_start + y_length / 2, y_label,  
            ha='right', va='center', fontsize=10)  

add_scale_bar(axs[1], x_length=0.001, y_length=0.04, x_label='1 ms', y_label='20 mV')

# axis limits for plots
axs[0].set_xlim(0.005, 0.015)
axs[1].set_xlim(0.005, 0.015)

plt.subplots_adjust(left=0.15, right=0.85, bottom=0.15, top=0.85)

plt.savefig('19_20_plots.png', dpi=1000, bbox_inches='tight')
plt.show()
