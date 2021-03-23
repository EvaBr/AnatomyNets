#%%
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd


def compare_curves(list_of_names, plot_names = None, individ_Dices = [3]):
    if plot_names==None:
        plot_names = list_of_names
    #read in metrics
    metrics = {name: pd.read_csv(f"RESULTS/{name}.csv") for name in list_of_names}
    Dice_names = ['Dice_bck','Dice_Bladder', 'Dice_KidneyL', 'Dice_Liver', 'Dice_Pancreas', 'Dice_Spleen', 'Dice_KidneyR']
    to_plot = ['Loss'] + [Dice_names[i] for i in individ_Dices]

    L = len(to_plot)
    cols, rows = 3, np.ceil(L/3)
    for tip in [("", "TRAINING"), ("val_", "VALIDATION")]:
        plt.figure(figsize = (cols*8,rows*5))
        plt.suptitle(tip[1])
        for idx, what in enumerate(to_plot):
            plt.subplot(rows, cols, idx+1)
            plt.title(what)
            for name in metrics:
                plt.plot(getattr(metrics[name], f"{tip[0]}{what}"), label=name)
            plt.legend()
        plt.show()
    
#%%
compare_curves(['First_unet', 'First_deepmed','First_pspnet'])
# %%
