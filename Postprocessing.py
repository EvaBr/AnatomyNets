#%%
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd
import Networks
import torch
import glob
from Slicing import flatten_one_hot
import matplotlib.patches as mpatches
import matplotlib
import random
#matplotlib.use('Agg')



def compare_curves(list_of_names, plot_names = None, individ_Dices = [3]):
    if plot_names==None:
        plot_names = list_of_names
    #read in metrics
    metrics = {name: pd.read_csv(f"RESULTS/{name}.csv") for name in list_of_names}
    Dice_names = ['Dice_bck','Dice_Bladder', 'Dice_KidneyL', 'Dice_Liver', 'Dice_Pancreas', 'Dice_Spleen', 'Dice_KidneyR']
    to_plot = ['Loss'] + [Dice_names[i] for i in individ_Dices]

    L = len(to_plot)
    cols, rows = min(3,L), np.ceil(L/3)
    for tip in [("", "TRAINING"), ("val_", "VALIDATION")]:
        plt.figure(figsize = (cols*7,rows*5))
        plt.suptitle(tip[1])
        for idx, what in enumerate(to_plot):
            plt.subplot(rows, cols, idx+1)
            plt.title(what)
            for name in metrics:
                plt.plot(getattr(metrics[name], f"{tip[0]}{what}"), label=name)
            plt.legend()
        plt.show()
    
#%%
#compare_curves(['Third_unet', 'Fourth_unet'])
# %%


def plotOutput(params, datafolder, pids, doeval=True):
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
    
    if doeval:
        net.eval()
       
    #load data&GT
    if isinstance(pids, str):
        pids = [pids]

    use_in2 = Arg['network']=='DeepMedic'
    allL = 0
    allfindgts = []
    allfindin1 = []
    allfindin2 = []
    for pid in pids:
        findgts = glob.glob(f"./{datafolder}/*/gt/*{pid}*.npy")
        findin1 = glob.glob(f"./{datafolder}/*/in1/*{pid}*.npy")
        findin2 = glob.glob(f"./{datafolder}/*/in2/*{pid}*.npy")
        findgts.sort(), findin1.sort(), findin2.sort()
        #all subslices in one image.
        ind = 1
        L = len(findgts)
        if  L>20: #ugly but needed to avoid too long compute
            take20 = random.sample(range(L), 20)
            findgts = [findgts[tk] for tk in take20]
            findin1 = [findin1[tk] for tk in take20]
            if use_in2:
                findin2 = [findin2[tk] for tk in take20]
            L=20
        
        allL += L 
        allfindgts.extend(findgts)
        allfindin1.extend(findin1)
        allfindin2.extend(findin2)

    
    organs = ['Bckg', 'Bladder', 'KidneyL', 'Liver', 'Pancreas', 'Spleen', 'KidneyR']
    if len(organs)!=Arg['n_class']: #in case not POEM dataset used
        organs = [str(zblj) for zblj in range(Arg['n_class'])]

    tgtonehot = np.load(allfindgts[0]).ndim>2 #are targets one hot encoded?

    data = [torch.stack([torch.from_numpy(np.load(i1)).float() for i1 in allfindin1], dim=0)]
    target = [flatten_one_hot(np.load(g)) if tgtonehot else np.load(g) for g in allfindgts] 
       
    if use_in2:
        in2 = torch.stack([torch.from_numpy(np.load(i2)).float() for i2 in allfindin2], dim=0)
        data.append(torch.from_numpy(in2[np.newaxis,:]).float())
    
    out = net(*data)
    outs = [flatten_one_hot(o.detach().squeeze().numpy()) for o in out] 
  # target, out = CenterCropTensor(target, out) #crop to be more comparable?

    fig = plt.figure(figsize=(10, allL*6)) 
    for ind in range(1, len(outs)+1):  
        #now plot :)
        plt.subplot(allL,2,2*ind-1)
        plt.title('GT')
        plt.axis('off')
        plt.imshow(target[ind-1], cmap='Spectral', vmin=0, vmax=Arg['n_class'])
        plt.subplot(allL,2,2*ind)
        plt.title('OUT')
        plt.axis('off')
        im = plt.imshow(outs[ind-1], cmap='Spectral', vmin=0, vmax=Arg['n_class'])
        
        values = np.arange(Arg['n_class'])
        colors = [ im.cmap(im.norm(value)) for value in values]
        # create a patch (proxy artist) for every color 
        patches = [ mpatches.Patch(color=colors[i], label=organs[i]) for i in range(len(values)) ]
        # put those patched as legend-handles into the legend
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1.), loc=2, borderaxespad=0. )

    plt.show()
    #plt.savefig('foo.png')

#%%
#plotOutput('unet', 'POEM', '500026_0')

# %%
#plotOutput('unet_batch8', 'POEM80', ['subj500357_28_0','subj500086_39_0', 'subj500124_69_0','subj500357_32_0',
#'subj500288_64_1', 'subj500348_31_0','subj500348_42_0',
#'subj500061_26_0'])