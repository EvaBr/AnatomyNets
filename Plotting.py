import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import re
from pathlib import Path


def plotSeg(pathtofajl, slicestuple=(90,35,130), t=0):
    #fajl = 'dm1dts_v4'
    #fil = 'RESULTS/OUTS/out' + str(subjnr) + '_'+fajl+'.npy'
    fil = pathtofajl
    subjnr = re.findall("500\d+", fil)[0]

    #which slices to plot:
    s1,s2,s3 = slicestuple 
    #bydim = 0 #set to 0 when in 3d

    img = np.load(fil)
    
    gt = list(Path('POEM', 'segms').glob(f'Cropped*{subjnr}*'))[0]
    gt = nib.load(str(gt)).get_fdata()
    mask = list(Path('POEM', 'masks').glob(f'cropped*{subjnr}*'))[0]
    mask = nib.load(str(mask)).get_fdata()
    print((gt.shape, img.shape))
    #gt, img = CenterCropTensor3d(gt, img)
    #
    #gx,gy,gz = gt.shape
    #img = img[:gx,:gy,:gz]
    
    sh1,sh2,sh3=gt.shape
    img = img[:sh1-t,:sh2-t,:sh3-t]
    gt = gt[t:,t:,t:]

    plt.figure(figsize=(10,10))
    plt.subplot(3,3,1)
    plt.imshow(img[:,:,s3].squeeze().T, extent=(0,100,0,150), vmin=0, vmax=7)
    plt.title('axial OUT')
    plt.axis('off')
    plt.subplot(3,3,2)
    plt.imshow(img[:,s2,:].squeeze(), vmin=0, vmax=7)
    plt.title('coronal OUT')
    plt.axis('off')
    plt.subplot(3,3,3)
    plt.imshow(img[s1,:,:].squeeze(), extent=(0,100,0,150), vmin=0, vmax=7)
    plt.title('saggital OUT')
    plt.subplot(3,3,4)
    plt.imshow(gt[:,:,s3].squeeze().T, extent=(0,100,0,150), vmin=0, vmax=7)
    plt.title('axial GT')
    plt.axis('off')
    plt.subplot(3,3,5)
    plt.imshow(gt[:,s2,:].squeeze(), vmin=0, vmax=7)
    plt.title('coronal GT')
    plt.axis('off')
    plt.subplot(3,3,6)
    plt.imshow(gt[s1,:,:].squeeze(), extent=(0,100,0,150), vmin=0, vmax=7)
    plt.title('sagital GT')
    plt.axis('off')


    newim = np.zeros((3,sh1-t,sh2-t,sh3-t))
    newim[0,...] = gt
    newim[1,...] = img

    plt.subplot(3,3,7)
    plt.imshow(newim[:,:,:,s3].squeeze().T, extent=(0,100,0,150), vmin=0, vmax=7)
    plt.title('axial GT')
    plt.axis('off')
    plt.subplot(3,3,8)
    plt.imshow(np.transpose(newim[:,:,s2,:].squeeze(),(1,2,0)), vmin=0, vmax=7)
    plt.title('coronal GT')
    plt.axis('off')
    plt.subplot(3,3,9)
    plt.imshow(np.transpose(newim[:,s1,:,:].squeeze(),(1,2,0)), extent=(0,100,0,150), vmin=0, vmax=7)
    plt.title('sagital GT')
    plt.axis('off')


#%%
def compare_curves(result_folder, list_of_names, epochs=0, plot_names = None, individ_Dices=[0,1,2,3,4,5,6]):
#        plot_names = kako jih imenovat v plotu. If none, reuse list_of_names
#        individ_dices = which individual dices to plot the curves for (if empty, plots only GDL and Loss"""
    #if epochs==0, we plot all, otherwise plot only psecified amount.

    if plot_names==None:
        plot_names = list_of_names
    #read in metrics
    pot = Path('RESULTS', result_folder)
    #metrics = {plotname: pd.read_csv(f"RESULTS/{name}.csv") for plotname,name in zip(plot_names, list_of_names)}
    metrics = {plotname: pd.read_csv(list(Path(pot, name).rglob('*.csv'))[0]) for plotname,name in zip(plot_names, list_of_names)}
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
                tmp = getattr(metrics[name], f"{tip[0]}{what}")
                if epochs>0:
                    tmp = tmp[:epochs]
                plt.plot(tmp, label=name)
            plt.legend()
        plt.show()


def calcMetrics(gt,img):
    #precrec = precision_recall_fscore_support(gt.flatten(), img.flatten())
    metrics = np.zeros((7,4))
    #print((gt.shape, img.shape))
    
    for clas in range(7):
        inter = ((gt==clas)*(img==clas)).sum() #(sliceinter==clas).sum()
    
        gtclas = (gt==clas).sum()
        imclas = (img==clas).sum()
        #recall
        tmp = inter/gtclas if gtclas>0 else np.nan
        metrics[clas, 1] = tmp
        #precision
        tmr = inter/imclas if imclas>0 else np.nan
        metrics[clas, 2] = tmr 
        #fscore
        metrics[clas, 3] = np.nan if (np.isnan(tmr) or np.isnan(tmp)) else 2*tmp*tmr/(tmp+tmr)
        #dice
        metrics[clas, 0] = 2*inter/(gtclas+imclas) if (gtclas>0 or imclas>0) else np.nan
    return metrics #, precrec
 
def getAllMetrics(fajl, subjnr, slicetuple=None): #(120,35,70)):
    fil = 'out' + str(subjnr) + '_'+fajl+'.npy'

    gt = list(Path('POEM', 'segms').glob(f'Cropped*{fil[3:9]}*'))[0]
    gt = nib.load(str(gt)).get_fdata()
    img = np.load(fil)

   # print((gt.shape, img.shape))

    #crop to same size <-- shouldn't be needed anymore, as saving is fixed
    #gt, img = CenterCropTensor3d(gt, img)
    ####
    #gx,gy,gz = gt.shape
    #img = img[:gx,:gy,:gz]
    ####

   # print((gt.shape, img.shape))
    if slicetuple==None:
        metrics = calcMetrics(gt, img)
        print('\n3D METRICS')
        print('         Dice,  Recall, Precision, F1: ')
        for i, lst in enumerate(metrics):
            print(f"class {i}: {lst}")
        return {'3D': metrics}

    s1, s2, s3 = slicetuple
    metrics1 = calcMetrics(gt[s1,:,:],img[s1,:,:])
    metrics2 = calcMetrics(gt[:,s2,:],img[:,s2,:])
    metrics3 = calcMetrics(gt[:,:,s3],img[:,:,s3])

    np.set_printoptions(precision=4)

    print('\nSAGGITAL SLICE')
    print('         Dice,  Recall, Precision, F1: ')
    for i, lst in enumerate(metrics1):
        print(f"class {i}: {lst}")


    print('\nCORONAL SLICE')
    print('         Dice, Recall, Precision, F1: ')
    for i, lst in enumerate(metrics2):
        print(f"class {i}: {lst}")

    print('\nAXIAL SLICE')
    print('         Dice, Recall, Precision, F1: ')
    for i, lst in enumerate(metrics3):
        print(f"class {i}: {lst}")

    return {'sagital': metrics1, 'coronal': metrics2, 'axial': metrics3}


# %%
