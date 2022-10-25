#%%
from Postprocessing import Compute3DDice, compare_curves, plotSeg
import numpy as np
from pathlib import Path
import nibabel as nib
from helpers import CenterCropTensor3d
import pandas as pd

#Compute3DDice(500017,'poem80_dts/unet_w/unet',80)
#Compute3DDice(500017,'poem80_dts/deepmed_w_dts/deepmed_dts',80)

#Compute3DDice(500017,'poem25-3D/deepmed/deepmed',25)
#Compute3DDice(500017,'poem25_all/pnet3d/pnet3d',25)

#when step=1, cuda is a must, it's slow anyhow
#Compute3DDice(500017,'poem25_new/unet/unet',25, batch=20, step=1, dev='cuda') 
#Compute3DDice(500017,'poem25_new/unet_dts/unet_dts',25, batch=20)
#Compute3DDice(500017,'poem25_new/unet_dts/unet_dts',25, dev='cuda', step=2)


#%%
mapa = 'poem'
konc = "_v4"
Compute3DDice(500062, mapa+'/deepmed1/deepmed', 25, batch=32, step=3, dev='cuda',saveout=True, savename='dm1'+konc)
Compute3DDice(500062, mapa+'/deepmed_dts1/deepmed', 25, batch=32, step=3, dev='cuda',saveout=True, savename='dm1dts'+konc)
Compute3DDice(500062, mapa+'/pnet1/pnet', 25, batch=32, step=3, dev='cuda',saveout=True,savename='pn1'+konc)
Compute3DDice(500062, mapa+'/pnet_dts1/pnet', 25, batch=32, step=3, dev='cuda',saveout=True, savename='pn1dts'+konc)
Compute3DDice(500062, mapa+'/unet1/unet', 25, batch=32, step=3, dev='cuda',saveout=True, savename='un1'+konc)
Compute3DDice(500062, mapa+'/unet_dts1/unet', 25, batch=32, step=3, dev='cuda',saveout=True, savename='un1dts'+konc)


#%%
Compute3DDice(500062, mapa+'/deepmed2/deepmed', 25, batch=32, bydim=2, step=3, dev='cuda', saveout=True, savename='dm2'+konc)
Compute3DDice(500062, mapa+'/deepmed_dts2/deepmed', 25, batch=32, bydim=2, step=3, dev='cuda', saveout=True, savename='dm2dts'+konc)
Compute3DDice(500062, mapa+'/pnet2/pnet', 25, batch=32, bydim=2, step=3, dev='cuda', saveout=True, savename='pn2'+konc)
Compute3DDice(500062, mapa+'/pnet_dts2/pnet', 25, batch=32, bydim=2, step=3, dev='cuda', saveout=True, savename='pn2dts'+konc)
Compute3DDice(500062, mapa+'/unet2/unet', 25, batch=32, bydim=2, step=3, dev='cuda', saveout=True, savename='un2'+konc)
Compute3DDice(500062, mapa+'/unet_dts2/unet', 25, batch=32, bydim=2, step=3, dev='cuda', saveout=True, savename='un2dts'+konc)


#%%
mapa = 'poem'
konc = '_v8'
Compute3DDice(500062, mapa+'/deepmed_dts3d/deepmed', 25, batch=32, step=3, dev='cuda', saveout=True, savename='dm3Ddts'+konc)
Compute3DDice(500062, mapa+'/deepmed3d/deepmed', 25, batch=32, step=3, dev='cuda', saveout=True, savename='dm3D'+konc)

#%%
Compute3DDice(500022, 'un3d_2_10/unet', 25, batch=32, step=4, dev='cuda', saveout=True, savename='un3d_2_best')
torch.cuda.empty_cache()
#%%
Compute3DDice(500022, 'un3d_dt_2_10/unet', 25, batch=32, step=4, dev='cuda', saveout=True, savename='un3d_dt_2_best')
torch.cuda.empty_cache()
#%%
Compute3DDice(500053, 'un3d_2_10/unet', 25, batch=32, step=4, dev='cuda', saveout=True, savename='un3d_2_best')
torch.cuda.empty_cache()
#%%
Compute3DDice(500053, 'un3d_dt_2_10/unet', 25, batch=32, step=4, dev='cuda', saveout=True, savename='un3d_dt_2_best')
torch.cuda.empty_cache()


#%% # check dices on a (set of) subject, given a training setting 
np.set_printoptions(precision=3)
fajl = 'un3d_dt'
#fajl = 'un3d_dt_10'
dcs = 'dices_'+fajl+'.csv'
dcs = pd.read_csv(dcs)
print(dcs)



#%%
#check training/Val curves
compare_curves("", ['un3d_10', 'un3d_dt_10'])


###########################################################
# %%
#get validation set class sizes for better sampling/loss settings:
from helpers import getpid

folder = 'POEM25'

allpids = np.unique([getpid(str(i)) for i in Path(folder, 'VAL', 'gt').glob('*.npy')])
allfiles = sorted([p for p in Path('POEM', 'segms').glob('*') if getpid(str(p)) in allpids])
allmasks = sorted([p for p in Path('POEM', 'masks').glob('*') if getpid(str(p)) in allpids])
N = len(allpids)
classes = np.zeros((7,))
for fil,mask in zip(allfiles, allmasks): 
    fajl = nib.load(str(fil)).get_fdata()
    maskca = nib.load(str(mask)).get_fdata()
    vsi = fajl[maskca==1]
    uniq, counts = np.unique(vsi, return_counts=True)
    classes[uniq.astype(int)] += counts

allinall = sum(classes)
avg_classes = classes/N
weighting = avg_classes/allinall

t=1/weighting
s=sum(t)
print(t/s) #[0.0004 , 0.217, 0.156, 0.016, 0.337, 0.113, 0.16 ]
#suggested weighting:
[0.01, 0.25, 0.16, 0.05, 0.34, 0.11, 0.16]
# %%
