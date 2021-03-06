#%%
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd
import Networks
import torch
import glob
from helpers import flatten_one_hot, get_one_hot, CenterCropTensor, CenterCropTensor3d
import matplotlib.patches as mpatches
import matplotlib
import random
from Losses import DicePerClass, AllDices, DicePerClassBinary, batchGDL
#matplotlib.use('Agg')




def compare_curves(list_of_names, plot_names = None, individ_Dices = [0,1,2,3,4,5,6]):
    if plot_names==None:
        plot_names = list_of_names
    #read in metrics
    metrics = {plotname: pd.read_csv(f"RESULTS/{name}.csv") for plotname,name in zip(plot_names, list_of_names)}
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
    

#compare_curves(['Third_unet', 'Fourth_unet'])

# %%
def getchn(args, string):
    cnt = 0
    whichchans = []
    start = [i for i in range(len(args)) if string in args[i]]
    if len(start)==0: #not given, using default channels 0,1
        return 2, [0,1]

    for i in range(start[0]+1, len(args)):
        if '--' in args[i]:
            break 
        cnt+=1
        whichchans.append(int(args[i]))
    return cnt, whichchans

def plotOutput(params, datafolder, pids, doeval=True, take20=None):
    """After training a net and saving its state to name PARAMS,
         run inference on subject PID from DATAFOLDER, and plot results+GT. 
         PID should be subject_slice, and all subject_slice_x will be run.
         E.g. plotOutput('First_unet', 'POEM', '500177_30'). """

    #default settings:
    Arg = {'network': None, 'n_class':7, 'in_channels':2, 'lower_in_channels':2, 'extractor_net':'resnet34'}
    
    with open(f"RESULTS/{params}_args.txt", "r") as ft:
        args = ft.read().splitlines()
    tmpargs = [i.strip('--').split("=") for i in args if ('--' in i and '=' in i)]
    chan1, whichin1 = getchn(args, 'in_chan')
    chan2, whichin2 = getchn(args, 'lower_in_chan')
    tmpargs += [['in_channels', chan1], ['lower_in_channels', chan2]]
    in3D='--in3D' in args
    args = dict(tmpargs)
    
    #overwrite if given in file:
    Arg.update(args)
    device = torch.device('cpu')

    net = getattr(Networks, Arg['network'])(Arg['in_channels'], Arg['n_class'], Arg['lower_in_channels'], Arg['extractor_net'], in3D)
    net = net.float()
    #now we can load learned params:
    loaded = torch.load(f"RESULTS/{params}", map_location=lambda storage, loc: storage)
    net.load_state_dict(loaded['state_dict'])
    
    if doeval:
        net.eval()
       
    net = net.to(device)
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
      #  print(findgts)

        #all subslices in one image.
        L = len(findgts)
        if  L>20: #ugly but needed to avoid too long compute
            if take20==None:
                take20 = random.sample(range(L), 20)
            findgts = [findgts[tk] for tk in take20]
            findin1 = [findin1[tk] for tk in take20]
            if use_in2:
                findin2 = [findin2[tk] for tk in take20]
            L=len(take20)
        
        allL += L 
        allfindgts.extend(findgts)
        allfindin1.extend(findin1)
        allfindin2.extend(findin2)

    
    organs = ['Bckg', 'Bladder', 'KidneyL', 'Liver', 'Pancreas', 'Spleen', 'KidneyR']
    if len(organs)!=Arg['n_class']: #in case not POEM dataset used
        organs = [str(zblj) for zblj in range(Arg['n_class'])]

    entmpgt = np.load(allfindgts[0])
    tgtonehot = entmpgt.shape[0]==7 #are targets one hot encoded?
    in3d = tgtonehot*(entmpgt.ndim==4) + (not tgtonehot)*(entmpgt.ndim==3)
    if in3d:
        #set the right function to use
        TensorCroping = CenterCropTensor3d
    else:
        TensorCroping = CenterCropTensor

    data = torch.stack([torch.from_numpy(np.load(i1)).float().to(device) for i1 in allfindin1], dim=0)
    data = [data[:, whichin1, ...]]
    target = [flatten_one_hot(np.load(g)) if tgtonehot else np.load(g) for g in allfindgts] 
    target_oh = torch.stack([torch.from_numpy(np.load(g)).to(device) if tgtonehot else torch.from_numpy(get_one_hot(np.load(g),7)).to(device) for g in allfindgts], dim=0)
       
    if use_in2:
        in2 = torch.stack([torch.from_numpy(np.load(i2)).float().to(device) for i2 in allfindin2], dim=0)
        data.append(in2[:, whichin2, ...])
    
    out = net(*data)
    target_oh, out = TensorCropping(target_oh, out)
    dices = AllDices(out, target_oh) #DicePerClass(out, target_oh)
  #  print((out.shape, target_oh.shape))
    outs = [flatten_one_hot(o.detach().squeeze().numpy()) for o in out] 
    
    fig, ax_tuple = plt.subplots(nrows=allL, ncols=2, figsize = (10, allL*6+1), tight_layout = True)
    #for compatibility reasons:
    if ax_tuple.ndim<2:
        ax_tuple = ax_tuple[np.newaxis, ...]
    plt.suptitle(params)
    for ind in range(len(outs)):  
        #now plot :)
        targetind, outsind = TensorCropping(target[ind], outs[ind]) #crop to be more comparable
      #  print((outsind.shape, targetind.shape))
        if in3d:
            sl = targetind.shape[-2]//2
            targetind, outsind = targetind[...,sl,:], outsind[...,sl,:]

        ax1 = ax_tuple[ind, 0]
        ax1.set_title('GT')
        ax1.axis('off')
        ax1.imshow(targetind, cmap='Spectral', vmin=0, vmax=Arg['n_class'])
        
        ax2 = ax_tuple[ind, 1]
        ax2.set_title('OUT')
        ax2.axis('off')
        im = ax2.imshow(outsind, cmap='Spectral', vmin=0, vmax=Arg['n_class'])
        
        values = np.arange(Arg['n_class'])
        colors = [ im.cmap(im.norm(value)) for value in values]
        # create a patch (proxy artist) for every color 
        patches = [ mpatches.Patch(color=colors[i], label=organs[i]) for i in range(len(values)) ]
        # put those patched as legend-handles into the legend
        ax2.legend(handles=patches, bbox_to_anchor=(1.05, 1.), loc=2, borderaxespad=0. )

        #write out also Dices:
        dajci = dices[ind].detach().squeeze().numpy()
        present_classes = [i for i in range(7) if i in target[ind]]
        t = ax2.text(1.08, 0.5, 'Dices:', size='medium', horizontalalignment='center', verticalalignment='center', transform=ax2.transAxes)
        for d in range(7): 
            t = ax2.text(1.1, 0.45-d*0.05, f"{organs[d]}: {dajci[d]:.3f}", size='small', transform=ax2.transAxes)

    plt.show()
    #plt.savefig('foo.png')
    #print(dices)
    return take20



# %%
# computing 3D Dice on entire subjects
from pathlib import Path
from tqdm import tqdm
import re
from natsort import natsorted 



def doInference(subject, folder, network, best_ep=True):
    """subject = sorted list of paths of all slices for a subject
        folder = results folder, from where your net will be loaded
        network = which net to load and use for eval
        best_ep = evaluate at best epoch? If false, it evals after complete training."""

    #now get network
    Arg = {'network': None, 'n_class':7, 'in_channels':2, 'lower_in_channels':2, 'extractor_net':'resnet34'}
    netfiles = Path("RESULTS", folder, network)
    argfiles = list(netfiles.glob("*_args.txt"))
    with open(argfiles[0], "r") as ft:
        args = ft.read().splitlines()
    tmpargs = [i.strip('--').split("=") for i in args if ('--' in i and '=' in i)]
    chan1, whichin1 = getchn(args, 'in_chan')
    chan2, whichin2 = getchn(args, 'lower_in_chan')
    tmpargs += [['in_channels', chan1], ['lower_in_channels', chan2]] 
    in3D = '--in3D' in args
    args = dict(tmpargs)
    
    #overwrite if given in file:
    Arg.update(args)
    use_in2 = Arg['network']=='DeepMedic'
    device = torch.device('cpu') #need speed now

    net = getattr(Networks, Arg['network'])(Arg['in_channels'], Arg['n_class'], Arg['lower_in_channels'], Arg['extractor_net'], in3D)
    net = net.float()
    #now we can load learned params:
    bestepfiles = list(netfiles.glob("*_bestepoch"))
    path_to_net = bestepfiles[0] if best_ep else Path(netfiles, network)
    loaded = torch.load(path_to_net, map_location=lambda storage, loc: storage)
    net.load_state_dict(loaded['state_dict'])
    net.eval()   
    net = net.to(device)

    #now do inference on entire subject simultaneously: but bcs of space limitations, needs cutting...
    #frst check if we're in 2D or 3D and change folder accordingly:
    data = torch.stack([torch.from_numpy(np.load(i1)).float() for i1 in subject], dim=0)
    data = [data[:, whichin1, ...]]
       
    if use_in2:
        subjekti2 = [Path(p.parents[1], 'in2', p.name) for p in subject]
        in2 = torch.stack([torch.from_numpy(np.load(i2)).float() for i2 in subjekti2], dim=0)
        data.append(in2[:, whichin2, ...])
    
    bigslice = 5
    out = []
    for start in range(0, len(data[0]), bigslice):
        in_batch = [datum[start:start+bigslice, ...].to(device) for datum in data]
       # print([i.shape for i in in_batch])
        tmp = net(*in_batch).detach().cpu()
        out.append(tmp)

    return torch.cat(out) #concatenate so it's like one big batch. 

def get3Ddice(PID, in3d, folder, network, best_ep=True):
    """PID = PID of a subject to calculate 3D Dice on 
        folder = results folder, from where your net will be loaded
        network = which net to load and use for eval
        best_ep = evaluate at best epoch? If false, it evals after complete training."""

    if in3d:
        subject = [i for i in Path('POEM_eval', 'TriD', 'in1').glob(f"*{PID}*.npy")]
    else:
        subject = [i for i in Path('POEM_eval', 'TwoD', 'in1').glob(f"*{PID}*.npy")]
       
    subject = natsorted(subject)
    #do inference to get batched output:
    out = doInference(subject, folder, network, best_ep)

    #calculate Dice:
    if in3d:
        gtpaths = natsorted(list(Path('POEM_eval', 'GTs_3D').glob(f"*{PID}*.npy")))
    else:
        gtpaths = natsorted(list(Path('POEM_eval', 'GTs_2D').glob(f"*{PID}*.npy")))
    gt = [torch.from_numpy(np.load(i)) for i in gtpaths]
    gt = torch.stack(gt)
    
   # print([[i.name, j.name] for i,j in zip(subject, gtpaths)])
    gt, _ = CenterCropTensor3d(gt, out) #in case of deepmedic, gt should be cropped to out size
    
    Dices = batchGDL(out.transpose(0,1), gt.transpose(0,1), binary=True) #DicePerClassBinary(out, gt) <-this avges over all patches... not cool

    return Dices 


def Calc3Ddices(folder, network, best_ep=True):
    """ folder = results folder, from where your net will be loaded
        network = which net to load and use for eval
        best_ep = evaluate at best epoch? If false, it evals after complete training."""
    #hardcoded for now. Use only subjects from eval. 
    PIDS = ['500026', '500061', '500075', '500117', '500242', '500268', '500288', '500291', 
            '500316', '500346', '500347', '500348', '500354', '500433', '500487']
    
    dices = [] 
    for pid in tqdm(PIDS):
        print(f"\nDoing PID {pid}...")
        dices.append(get3Ddice(pid, folder, network, best_ep).cpu().numpy())

    #save results in csv
    zaDF = np.stack([PIDS, np.vstack(dices)], axis=0)
    cols = ['PID', 'Dice_bck','Dice_Bladder', 'Dice_KidneyL', 'Dice_Liver', 'Dice_Pancreas', 'Dice_Spleen', 'Dice_KidneyR']
    df = pd.DataFrame(zaDF, columns=cols)
    
    with open(f'POEM_eval_{folder}_{network}.csv', 'w') as f:
        df.to_csv(f)  #OBS if file exists, it will only append data. 
    


    
# %%
