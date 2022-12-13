#%%
#from Postprocessing import Compute3DDice, compare_curves, plotSeg
from matplotlib.pyplot import title
from builtins import range
import numpy as np
from pathlib import Path
import nibabel as nib
import pandas as pd
from Plotting import compare_curves



compare_curves('VanillaCNN/rep1', ['van_3f', 'van_3f_dt']) #,epochs=60)


#%%
#test = {'1': [500051, 500022, 500379, 500167, 500327, 500117, 500297, 500077, 500280, 500253], 
#        '2': [500433, 500179, 500159, 500158, 500487, 500268, 500086, 500075, 500281, 500056], 
#        '3': [500235, 500242, 500017, 500204, 500358, 500357, 500348, 500346, 500304, 500473], 
#        '4': [500429, 500018, 500062, 500403, 500061, 500347, 500354, 500316, 500321, 500288], 
#        '5': [500291, 500424, 500406, 500053, 500026, 500241, 500205, 500395, 500318, 500124]}

#patchsize = 25

#for fold in test:
  #  torch.cuda.empty_cache()
  #  name = f"van_{fold}f/vanilla"
  #  print(name)
  #  a = Compute3DDice(test[fold], name, patchsize, 
  #          best='_80', step=10, saveout=False, savename='van_5fold')
  #  torch.cuda.empty_cache()
  #  print(a)

#    name = f"/van_{fold}f_dt/vanilla"
#    print(name)
#    torch.cuda.empty_cache()
#    a = Compute3DDice(test[fold], name, patchsize, 
#            best='_80', step=10, saveout=False, savename='van_5fold_dt_rep2')
#    torch.cuda.empty_cache()
#    print(a)


#%%
#calc averages and stds from saved resutls:
net5fold = pd.read_csv('RESULTS/OUTS/dices_van_5fold_rep2.csv',header=0, index_col=0, skiprows=lambda x: x>0 and x%11==0)
means = net5fold.mean()
stds = net5fold.std(ddof=0)

print('VANILLA NET')
print(", ".join([f"{m:.2f}(+-{s:.2f})" for m,s in zip(means[1:],stds[1:])]))



net5fold_dt = pd.read_csv('RESULTS/OUTS/van-5fold/dices_van_5fold_rep2_dt.csv',header=0, index_col=0, skiprows=lambda x: x>0 and x%11==0)
means_dt = net5fold_dt.mean()
stds_dt = net5fold_dt.std(ddof=0)


print('VANILLA NET + dt')
print(", ".join([f"{m:.2f}(+-{s:.2f})" for m,s in zip(means_dt[1:],stds_dt[1:])]))



#%%
import seaborn as sns

#violin plots of average dices
REPS = 2
NETS=['van'] #, 'un'] #, 'dm']
L=len(NETS)
netnames = {'van': 'Vanilla CNN', 'un': 'U-Net', 'dm': 'DeepMedic'}

alla = pd.concat([pd.read_csv(f'RESULTS/OUTS/{net}-5fold/dices_{net}_5fold_rep{rep+1}.csv',
        header=0, index_col=0, skiprows=lambda x: x>0 and x%11==0) for rep in range(REPS) for net in NETS] + 
        [pd.read_csv(f'RESULTS/OUTS/{net}-5fold/dices_{net}_5fold_dt_rep{rep+1}.csv',
        header=0, index_col=0, skiprows=lambda x: x>0 and x%11==0) for rep in range(REPS) for net in NETS], 
        axis=0, ignore_index=True)

nets = []
for i in NETS:
    nets.extend([netnames[i]]*50*REPS)

dices = alla.drop('bckg', axis=1).mean(axis=1)
otherdata = pd.DataFrame({'network': nets*2, # ['vanilla']*100 + ['U-net']*100,
                        'type':['original']*50*REPS*L+['with DTs']*50*REPS*L})
data = pd.concat([dices, otherdata], axis=1)
data.rename(columns={0:'subject Dices'}, inplace=True)

sns.violinplot(data=data, x="network", y="subject Dices", hue="type", split=True, title="*Unweighted")

#now weighted Dice:
weights = pd.Series([0, 1, 1, 1, 1, 1, 1])
dices = alla.apply(lambda x: np.asarray(x) * np.asarray(weights), axis=1)
data = pd.concat([dices, otherdata], axis=1)
data.rename(columns={0:'subject Dices'}, inplace=True)

sns.violinplot(data=data, x="network", y="subject Dices", hue="type", split=True, title="Weighted")


#%%
#violin plots per network for each organ individually
#
REPS = 5
net='van'

orig = pd.concat([pd.read_csv(f'RESULTS/OUTS/{net}-5fold/dices_{net}_5fold_rep{rep+1}.csv',
    header=0, index_col=0, skiprows=lambda x: x>0 and x%11==0) for rep in range(REPS)], axis=0, ignore_index=True)

withDt = pd.concat([pd.read_csv(f'RESULTS/OUTS/{net}-5fold/dices_{net}_5fold_dt_rep{rep+1}.csv',
    header=0, index_col=0, skiprows=lambda x: x>0 and x%11==0) for rep in range(REPS)], axis=0, ignore_index=True)


names=[]
for i in orig.columns.values.tolist()[1:]:
    names.extend([i]*50*REPS)

names = names*2
conV = pd.concat([orig.iloc[:,i] for i in range(1,7)] + [withDt.iloc[:,i] for i in range(1,7)], axis=0, ignore_index=True)
otherdata = pd.DataFrame({'class': names, 'type': ['original']*6*50*REPS + ['with DTs']*6*50*REPS})
alldata = pd.concat([conV, otherdata], axis=1)
alldata.rename(columns={0:'Dice'}, inplace=True)

sns.violinplot(data=alldata, x="class", y="Dice", hue="type", split=True)



#%%
from scipy.stats import wilcoxon, ttest_rel, rankdata
#un5fold_dt = pd.read_csv('RESULTS/OUTS/dices_van_5fold_dt.csv',header=0, index_col=0, skiprows=lambda x: x>0 and x%11==0)
#un5foldB_dt = pd.read_csv('RESULTS/OUTS/dices_unB_5fold_dt.csv',header=0, index_col=0, skiprows=lambda x: x>0 and x%11==0)
#un5fold = pd.read_csv('RESULTS/OUTS/dices_van_5fold.csv',header=0, index_col=0, skiprows=lambda x: x>0 and x%11==0)
#un5foldB = pd.read_csv('RESULTS/OUTS/dices_unB_5fold.csv',header=0, index_col=0, skiprows=lambda x: x>0 and x%11==0)

REPS = 5
net = 'van'
allreps = pd.concat([pd.read_csv(f'RESULTS/OUTS/{net}-5fold/dices_{net}_5fold_rep{rep+1}.csv',
    header=0, index_col=0, skiprows=lambda x: x>0 and x%11==0) for rep in range(REPS)], axis=1)
allrepsDT = pd.concat([pd.read_csv(f'RESULTS/OUTS/{net}-5fold/dices_{net}_5fold_dt_rep{rep+1}.csv',
    header=0, index_col=0, skiprows=lambda x: x>0 and x%11==0) for rep in range(REPS)], axis=1)

colnames = ['bckg', 'bladder', 'kidney1', 'liver', 'pancreas', 'spleen', 'kidney2']
repts = [f'rep{i+1}' for i in range(REPS)]
allreps.columns = pd.MultiIndex.from_product([repts, colnames], names=['reps', 'class'])
allrepsDT.columns = pd.MultiIndex.from_product([repts, colnames], names=['reps', 'class'])
#Calc averages over repetition dimension
tojoin = []
means = []
stds = []
stds2 = []
for col in colnames:
    tmp = allreps.xs(col, level='class', axis=1) #nr subj X nr reps: given class dices
    means.append(tmp.mean().mean())
    stds.append(np.asarray(tmp).std())
    tojoin.append(tmp.mean(axis=1))
    stds2.append(tojoin[-1].std(ddof=0))
meantable = pd.concat(tojoin,  axis=1)

tojoin_dt = []
means_dt = []
stds_dt = []
stds2_dt = []
for col in colnames:
    tmp = allrepsDT.xs(col, level='class', axis=1) #nr subj X nr reps: given class dices
    means_dt.append(tmp.mean().mean())
    stds_dt.append(np.asarray(tmp).std())
    tojoin_dt.append(tmp.mean(axis=1))
    stds2_dt.append(tojoin_dt[-1].std(ddof=0))
meantable_dt = pd.concat(tojoin_dt,  axis=1)


print(f"{net.upper()} NET")
print(", ".join([f"{m:.2f}(+-{s:.2f}, mean std +-{s2:.2f})" for m,s,s2 in zip(means[1:],stds[1:],stds2[1:])]))

print(f"{net.upper()} NET + dt")
print(", ".join([f"{m:.2f}(+-{s:.2f}, mean std +-{s2:.2f})" for m,s,s2 in zip(means_dt[1:],stds_dt[1:],stds2_dt[1:])]))




#%%
expectedBigger = meantable_dt #un5fold_dt 
baseline = meantable #un5fold 

for klas in range(7):
    a=expectedBigger.iloc[:, klas]
    b=baseline.iloc[:, klas]
    d = a - b
    #alternative would be:
        #resg = ttest_rel(a,b, alternative='greater')
        #res = ttest_rel(a,b)
    #but requires assumption on normlaity. while wilcoxon is nonparam.
    resg = wilcoxon(d, alternative='greater')
   # res = wilcoxon(d)
    print(d.name)
   # print('double sided: ')
   # print(res.statistic, res.pvalue)
   # print('greater: ')
    print(resg.statistic, resg.pvalue)
    
    #measuring effect size:
    ranks = rankdata(abs(d))
    S = sum(ranks)
    fave = sum(ranks[d>0])
    unfave = sum(ranks[d<=0])
    matched_pair_rank_biserial_corr = (fave - unfave)/S
    print(f"effect size: {matched_pair_rank_biserial_corr}")
    print()

#%%
#measuring effect size (by matched-pairs rank-biserial correlation,
# see https://journals.sagepub.com/doi/full/10.2466/11.IT.3.1):
for klas in range(7):
    a=expectedBigger.iloc[:, klas]
    b=baseline.iloc[:, klas]
    d = a-b 
    ranks = rankdata(abs(d))
    S = sum(ranks)
    fave = sum(ranks[d>0])
    unfave = sum(ranks[d<=0])
    matched_pair_rank_biserial_corr = (fave - unfave)/S

    print(a.name)
    print(matched_pair_rank_biserial_corr)
    print()



 










#%%
from Plotting import compare_curves
#check training/Val curves
compare_curves("", ['un3d_10', 'un3d_dt_10'])


###########################################################
# %%
#get validation set class sizes for better sampling/loss settings:
from helpers import getpid

folder = 'POEM'

#allpids = np.unique([getpid(str(i)) for i in Path(folder, 'VAL', 'gt').glob('*.npy')])

allpids = np.unique([getpid(str(i)) for i in Path(folder, 'masks').glob('*.nii')])
allfiles = sorted([p for p in Path('POEM', 'segms').glob('*') if getpid(str(p)) in allpids])
allfats = sorted([p for p in Path('POEM', 'watfat').glob('*fat_content.nii') if getpid(str(p)) in allpids])
allwats = sorted([p for p in Path('POEM', 'watfat').glob('*wat_content.nii') if getpid(str(p)) in allpids])
allmasks = sorted([p for p in Path('POEM', 'masks').glob('*') if getpid(str(p)) in allpids])
alldtx = sorted([p for p in Path('POEM', 'distmaps').glob('*_x.nii') if getpid(str(p)) in allpids])
alldty = sorted([p for p in Path('POEM', 'distmaps').glob('*_y.nii') if getpid(str(p)) in allpids])


N = len(allpids)
shapes=[]
maxmins=[]
imgranges=[]
classes = np.zeros((7,))
for fil,fat,wat,mask,dx,dy in zip(allfiles, allfats, allwats, allmasks, alldtx, alldty): 
    fajl = nib.load(str(fil)).get_fdata()
    fat = nib.load(str(fat)).get_fdata()
    wat = nib.load(str(wat)).get_fdata()
    maskca = nib.load(str(mask)).get_fdata()
    dtx = nib.load(str(dx)).get_fdata()
    dty = nib.load(str(dy)).get_fdata()
    vsi = fajl[maskca==1]
    uniq, counts = np.unique(vsi, return_counts=True)
    classes[uniq.astype(int)] += counts
    shapes.append(fajl.shape)
    maxmins.append([dtx.min(), dtx.max(), dty.min(), dty.max()])
    imgranges.append([min(wat.min(),fat.min()), max(wat.max(), fat.max())])

allinall = sum(classes)
avg_classes = classes/N
weighting = avg_classes/allinall

t=1/weighting
s=sum(t)
print(t/s) #[0.0004 , 0.217, 0.156, 0.016, 0.337, 0.113, 0.16 ]
#suggested weighting:
#[0.01, 0.25, 0.16, 0.05, 0.34, 0.11, 0.16]

print(np.unique(shapes, axis=0))
print(np.unique(maxmins, axis=0))
# %%
import matplotlib.pyplot as plt

PID1 = '500433' #y size 98
PID2 = '500281' #y size 80

fatt = nib.load(str(Path('POEM','watfat','cropped'+PID1+'_fat_content.nii'))).get_fdata()
fx = nib.load(str(Path('POEM','distmaps','distmap_'+PID1+'.nii_x.nii'))).get_fdata()
fy = nib.load(str(Path('POEM','distmaps','distmap_'+PID1+'.nii_y.nii'))).get_fdata()

skinny = nib.load(str(Path('POEM','watfat','cropped'+PID2+'_fat_content.nii'))).get_fdata()
sx = nib.load(str(Path('POEM','distmaps','distmap_'+PID2+'.nii_x.nii'))).get_fdata()
sy = nib.load(str(Path('POEM','distmaps','distmap_'+PID2+'.nii_y.nii'))).get_fdata()

sls1 = 50
sls2 = 40
plt.figure()
plt.subplot(2,3,1)
plt.imshow(fatt[:,sls1,:].squeeze())
plt.subplot(2,3,2)
plt.imshow(fx[:,sls1,:].squeeze())
plt.subplot(2,3,3)
plt.imshow(fy[:,sls1,:].squeeze())

plt.subplot(2,3,4)
plt.imshow(skinny[:,sls2,:].squeeze())
plt.subplot(2,3,5)
plt.imshow(sx[:,sls2,:].squeeze())
plt.subplot(2,3,6)
plt.imshow(sy[:,sls2,:].squeeze())
#%%


fmask= nib.load(str(Path('POEM','masks','cropped'+PID1+'_mask.nii'))).get_fdata()
smask = nib.load(str(Path('POEM','masks','cropped'+PID2+'_mask.nii'))).get_fdata()

#%%
tmpF = fatt[:,:,140].squeeze()
tmpFx = fx[:,:,140].squeeze()
tmpFy = fy[:,:,140].squeeze()
fgray = np.stack([tmpF, tmpF, tmpF],axis=-1)
tmpS = skinny[:,:,150].squeeze()
tmpSx = sx[:,:,150].squeeze()
tmpSy = sy[:,:,150].squeeze()
sgray = np.stack([tmpS, tmpS, tmpS],axis=-1)

#%%
#LMf[177:185, 189:197]=1 #,134:136] = 1
#LMf[73:81, 189:197]=1 #,135:137] = 1
LMfx = np.array([ 92, 176,  41, 209])/2  #[181, 77, 92, 176,  92, 160, 41, 209]
LMfy = (np.array([ 136, 136, 59, 57]) -57)*2 #[193, 193, 136, 136, 242, 252, 59, 57]
Fcenter = (LMfy.mean(), LMfx.mean())


LMsx = np.array([95, 173, 64, 204])/2
LMsy = (np.array([125, 124, 60,61])-60)*2
Scenter = (LMsy.mean(), LMsx.mean())


plt.figure()
plt.subplot(1,2,1)
plt.imshow(fgray, extent=(0,196,0,128))
plt.plot(LMfy, LMfx-4, 'rx')
plt.plot(*Fcenter, 'go')
plt.subplot(1,2,2)
plt.imshow(sgray, extent=(0,160,0,128))
plt.plot(LMsy,LMsx-4, 'rx')
plt.plot(*Scenter, 'go')
# %%

szx = 7
szy = 7
Fy,Fx = int(Fcenter[0]/2), int(Fcenter[1]*2)
Sy,Sx = int(Scenter[0]/2), int(Scenter[1]*2)

ttFx = tmpFx[Fx-szx:Fx+szx, Fy-szy:Fy+szy]
ttSx = tmpSx[Sx-szx:Sx+szx, Sy-szy:Sy+szy]
miniX = min(ttFx.min(), ttSx.min())
maxiX = max(ttFx.max(), ttSx.max())

ttFy = tmpFy[Fx-szx:Fx+szx, Fy-szy:Fy+szy]
ttSy = tmpSy[Sx-szx:Sx+szx, Sy-szy:Sy+szy]
miniY = min(ttFy.min(), ttSy.min())
maxiY = max(ttFy.max(), ttSy.max())


plt.figure()
plt.subplot(2,3,1)
plt.imshow(tmpF[(Fx-szx):(Fx+szx), (Fy-szy):(Fy+szy)])
plt.subplot(2,3,2)
plt.imshow(tmpFx[Fx-szx:Fx+szx, Fy-szy:Fy+szy], vmin=miniX, vmax=maxiX)
plt.subplot(2,3,3)
plt.imshow(tmpFy[Fx-szx:Fx+szx, Fy-szy:Fy+szy], vmin=miniY, vmax=maxiY)
plt.subplot(2,3,4)
plt.imshow(tmpS[Sx-szx:Sx+szx, Sy-szy:Sy+szy])
plt.subplot(2,3,5)
plt.imshow(tmpSx[Sx-szx:Sx+szx, Sy-szy:Sy+szy], vmin=miniX, vmax=maxiX)
plt.subplot(2,3,6)
plt.imshow(tmpSy[Sx-szx:Sx+szx, Sy-szy:Sy+szy], vmin=miniY, vmax=maxiY)
# %%
plt.figure()
plt.subplot(1,2,1)
plt.imshow(ttSy+ttSx, vmin=min(miniX,miniY), vmax=maxiX+maxiY)

plt.subplot(1,2,2)
plt.imshow(ttFy+ttFx, vmin=min(miniX,miniY), vmax=maxiX+maxiY)
# %%
