from Postprocessing import Compute3DDice

#Compute3DDice(500017,'poem80_dts/unet_w/unet',80)
#Compute3DDice(500017,'poem80_dts/deepmed_w_dts/deepmed_dts',80)

#Compute3DDice(500017,'poem25-3D/deepmed/deepmed',25)
#Compute3DDice(500017,'poem25_all/pnet3d/pnet3d',25)

#when step=1, cuda is a must, it's slow anyhow
#Compute3DDice(500017,'poem25_new/unet/unet',25, batch=20, step=1, dev='cuda') 
#Compute3DDice(500017,'poem25_new/unet_dts/unet_dts',25, batch=20)
#Compute3DDice(500017,'poem25_new/unet_dts/unet_dts',25, dev='cuda', step=2)

#%%
Compute3DDice('500062', 'poem25/deepmed1/deepmed', 25, batch=20, step=3, dev='cuda')
Compute3DDice('500062', 'poem25/deepmed_dts1/deepmed', 25, batch=20, step=3, dev='cuda')
Compute3DDice('500062', 'poem25/pnet1/pnet', 25, batch=20, step=3, dev='cuda')
Compute3DDice('500062', 'poem25/pnet_dts1/pnet', 25, batch=20, step=3, dev='cuda')
Compute3DDice('500062', 'poem25/unet1/unet', 25, batch=20, step=3, dev='cuda')
Compute3DDice('500062', 'poem25/unet_dts1/unet', 25, batch=20, step=3, dev='cuda')


#%%
Compute3DDice('500062', 'poem25/deepmed2/deepmed', 25, batch=20, bydim=2, step=3, dev='cuda')
Compute3DDice('500062', 'poem25/deepmed_dts2/deepmed', 25, batch=20, bydim=2, step=3, dev='cuda')
Compute3DDice('500062', 'poem25/pnet2/pnet', 25, batch=20, bydim=2, step=3, dev='cuda')
Compute3DDice('500062', 'poem25/pnet_dts2/pnet', 25, batch=20, bydim=2, step=3, dev='cuda')
Compute3DDice('500062', 'poem25/unet2/unet', 25, batch=20, bydim=2, step=3, dev='cuda')
Compute3DDice('500062', 'poem25/unet_dts2/unet', 25, batch=20, bydim=2, step=3, dev='cuda')


#%%
Compute3DDice('500062', 'poem25/deepmed3d/deepmed', 25, batch=10, step=3, dev='cuda')
Compute3DDice('500062', 'poem25/deepmed_dts3d/deepmed', 25, batch=10, step=3, dev='cuda')
Compute3DDice('500062', 'poem25/pnet3d/pnet', 25, batch=10, step=3, dev='cuda')
Compute3DDice('500062', 'poem25/pnet_dts3d/pnet', 25, batch=10, step=3, dev='cuda')
Compute3DDice('500062', 'poem25/unet3d/unet', 25, batch=10, step=3, dev='cuda')
Compute3DDice('500062', 'poem25/unet_dts3d/unet', 25, batch=10, step=3, dev='cuda')