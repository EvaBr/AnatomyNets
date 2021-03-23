#%%
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd
import Networks
import torch
import glob
from Slicing import flatten_one_hot
import matplotlib.patches as mpatches


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

def plotOutput(params, datafolder, pid):
    """After training a net and saving its state to name PARAMS,
         run inference on subject PID from DATAFOLDER, and plot results+GT. 
         PID should be subject_slice, and all subject_slice_x will be run.
         E.g. plotOutput('First_unet', 'POEM', '500177_30'). """

    #default settings:
    Arg = {'network': None, 'n_class':7, 'in_channels':2, 'lower_in_channels':2, 'extractor_net':'resnet34'}
    
    with open(f"RESULTS/{params}_args.txt", "r") as ft:
        args = ft.read().splitlines()
    args = dict([i.strip('--').split("=") for i in args if ("#" not in i and 'debug' not in i)])
    
    #overwrite if given in file:
    Arg.update(args)

    net = getattr(Networks, Arg['network'])(Arg['in_channels'], Arg['n_class'], Arg['lower_in_channels'], Arg['extractor_net'])
    net = net.float()
    #now we can load learned params:
    loaded = torch.load(f"RESULTS/{params}")
    net.load_state_dict(loaded['state_dict'])
    net.eval()
       
    #load data&GT
    use_in2 = Arg['network']=='DeepMedic'
    findgts = glob.glob(f"./{datafolder}/*/gt/subj{pid}_*.npy")
    findin1 = glob.glob(f"./{datafolder}/*/in1/subj{pid}_*.npy")
    findin2 = glob.glob(f"./{datafolder}/*/in2/subj{pid}_*.npy")
    findgts.sort(), findin1.sort(), findin2.sort()
    #all subslices in one image.
    ind = 1
    L = len(findgts)
    fig = plt.figure(figsize=(10, L*6))
    axes = []
    organs = ['Bckg', 'Bladder', 'KidneyL', 'Liver', 'Pancreas', 'Spleen', 'KidneyR']
    for g,i1,i2 in zip(findgts, findin1, findin2):
        in1 = np.load(i1)
        data = [torch.from_numpy(in1[np.newaxis,:]).float()]
        in2 = np.load(i2)
        target = flatten_one_hot(np.load(g))
        if use_in2:
            data.append(torch.from_numpy(in2[np.newaxis,:]).float())
    
        out = net(*data)
  #      target, out = CenterCropTensor(target, out) #crop to be more comparable?

        #now plot :)
        plt.subplot(L,2,ind)
        plt.title('GT')
        plt.imshow(target, cmap='Spectral', vmin=0, vmax=7)
        plt.subplot(L,2,ind+1)
        plt.title('OUT')
        im = plt.imshow(flatten_one_hot(np.squeeze(out.detach().numpy())), cmap='Spectral', vmin=0, vmax=7)
        ind = ind+2
        
        values = np.arange(7)
        colors = [ im.cmap(im.norm(value)) for value in values]
        # create a patch (proxy artist) for every color 
        patches = [ mpatches.Patch(color=colors[i], label=organs[i]) for i in range(len(values)) ]
        # put those patched as legend-handles into the legend
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1.), loc=2, borderaxespad=0. )

    plt.show()

            


# %%
