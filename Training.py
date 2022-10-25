import Networks
from Losses import MultiTaskLoss, DicePerClassBinary, batchGDL
from Dataloaders import get_loaders
import argparse
import torch
from torch import optim, nn
from tqdm import tqdm
import numpy as np
import pandas as pd
import sys
import os
from helpers import CenterCropTensor, CenterCropTensor3d, flatten_one_hot
import matplotlib.pyplot as plt 

np.set_printoptions(precision=3, suppress=True)
torch.set_printoptions(precision=3, sci_mode=False)


def save_run(train_m, val_m, model, optimizer, save_to, epoch):
    """Saves the final state of network for re-loading, and CSV of metrics, and a txt file of args used for the run."""

    if not os.path.exists('RESULTS'):
        os.mkdir('RESULTS')
    joint_dic = {**train_m, **val_m}
    df = pd.DataFrame({k: v.cpu().numpy().tolist() for k,v in joint_dic.items()})
    cols = ['Dice_bck','Dice_Bladder', 'Dice_KidneyL', 'Dice_Liver', 'Dice_Pancreas', 'Dice_Spleen', 'Dice_KidneyR']
    cols_v = [f"val_{c}" for c in cols]
    df = pd.concat([df['Loss'], df['GDL'], df['GDLbin'], pd.DataFrame(df['Dice'].values.tolist(), columns=cols), 
                df['val_Loss'], df['val_GDL'], df['val_GDLbin'], pd.DataFrame(df['val_Dice'].values.tolist(), columns=cols_v)], axis=1)
    with open(f'RESULTS/{save_to}.csv', 'a') as f:
        df.to_csv(f, header=f.tell()==0)  #OBS if file exists, it will only append data. 
    with open(f'RESULTS/{save_to}_args.txt', 'a') as f:
        f.write('\n'.join(sys.argv[1:]))
        f.write('\n########################\n') #to separate args of diff. runs, if net reloaded and rerun

    torch.save({'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()}, f'RESULTS/{save_to}')


def setup(args: argparse.Namespace):
    device = torch.device("cpu") if (not torch.cuda.is_available() or args.cpu) else torch.device("cuda")

    net = getattr(Networks, args.network)(len(args.in_channels), args.n_class, in_channels_lower=len(args.lower_in_channels), extractor_net=args.extractor_net, TriD=args.in3D)
    net = net.float()
    to_init = Networks.weights_init(args.network)
    net.apply(to_init)

    start_epoch = 0
    optimizer =  optim.Adam(net.parameters(), lr=args.l_rate, betas=(0.9, 0.99), amsgrad=False)
    if args.restore_from!="":
        #restore pretrained and saved network:
        loaded = torch.load(args.restore_from)
        #assert loaded['name'] == args.network
        net.load_state_dict(loaded['state_dict'])
        optimizer.load_state_dict(loaded['optimizer'])
        start_epoch = loaded['epoch']+1

        # now individually transfer the optimizer parts...
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
    
    model = net.to(device)
    loss = MultiTaskLoss(args.losses, device, in3d=args.in3D)
    train_loader, val_loader = get_loaders(args.network, args.dataset, args.n_class, args.in_channels, args.lower_in_channels, args.batch_size, args.debug)

    return model, optimizer, loss, train_loader, val_loader, device, start_epoch



def train(args: argparse.Namespace):
    torch.autograd.set_detect_anomaly(True)
    lr: float = args.l_rate
    is3d: bool = args.in3D
    model, optimizer, loss_fn, train_loader, val_loader, device, start_epoch = setup(args)

    train_metrics = {"Loss": torch.zeros((args.n_epoch, ), device=device).type(torch.float32), 
                     "Dice": torch.zeros((args.n_epoch, args.n_class), device=device).type(torch.float32),
                     "GDL": torch.zeros((args.n_epoch,), device=device).type(torch.float32), 
                     "GDLbin": torch.zeros((args.n_epoch,), device=device).type(torch.float32)}
    val_metrics = {"val_Loss": torch.zeros((args.n_epoch, ), device=device).type(torch.float32), 
                   "val_Dice": torch.zeros((args.n_epoch, args.n_class), device=device).type(torch.float32),
                   "val_GDL": torch.zeros((args.n_epoch,), device=device).type(torch.float32), 
                   "val_GDLbin": torch.zeros((args.n_epoch,), device=device).type(torch.float32)}

    best_epoch: int = start_epoch
    best_avg_dice: float = 0
    
    numim = len(train_loader.dataset)
    numimval = len(val_loader.dataset)
    TensorCenterCropping = CenterCropTensor #assume 2d from the beginning
    if is3d:
        TensorCenterCropping = CenterCropTensor3d
            
    for epoch in range(start_epoch, args.n_epoch+start_epoch):
        #train
        model.train()
        NN = len(train_loader)
        train_iterator = tqdm(train_loader, ncols=120, total=NN, leave=False)
        
        epoch_train_metrics = {"Loss": torch.tensor(0.0, device=device), 
                            "Dice": torch.zeros((7,), device=device).type(torch.float32)}
        GDL: float = 0
        GDLbin: float = 0
        for data_tuple in train_iterator:
            optimizer.zero_grad()
            data, target = [i.to(device) for i in data_tuple[:-1]], data_tuple[-1].to(device)
         #   print([i.shape for i in data])
            out = model(*data)

         #   print(f"data0: {data[0].shape}, target: {target.shape}, out: {out.shape}")
            #for some nets, output will be smaller than target. Crop target centrally to match output:
            target, out = TensorCenterCropping(target, out)
            loss = loss_fn(out, target)

            loss.backward()
            optimizer.step()

            dice = DicePerClassBinary(out.detach(), target.detach())
            GDL += batchGDL(out.detach(), target.detach()).sum()
            GDLbin += batchGDL(out.detach(), target.detach(), True).sum()
            epoch_train_metrics["Loss"] += loss.item()
            epoch_train_metrics["Dice"] += dice.detach()

            status = {"loss": loss.item(), "Dice": dice.detach().cpu().numpy()} #is there a way to print nicely without copying to cpu?
            train_iterator.set_postfix(status) #description(status)
        
           
        if args.debug and epoch%5==0:
            plt.figure()
            elstr = "...]"
            if is3d:
                elstr = f"{target.shape[-2]//2},...]"

            for elem in range(1, args.batch_size+1):  #in case of 3d data, only plot middle slice
                plt.subplot(args.batch_size,2,2*elem-1)
                plt.imshow(flatten_one_hot(eval("target[elem-1,"+elstr+".detach().squeeze().cpu().numpy()")),  cmap='Spectral', vmin=0,vmax=7)
                plt.subplot(args.batch_size,2,2*elem)
                plt.imshow(flatten_one_hot(eval("out[elem-1,"+elstr+".detach().squeeze().cpu().numpy()")),  cmap='Spectral', vmin=0,vmax=7)
            plt.show()

        
        #save results:
        for i in ['Loss', 'Dice']: #train_metrics:
            train_metrics[i][epoch-start_epoch, ...] = epoch_train_metrics[i]/NN
        print(f"EPOCH {epoch}: \n [TRAIN] Loss={train_metrics['Loss'][epoch-start_epoch]}, GDL={GDL/numim}, Dice={train_metrics['Dice'][epoch-start_epoch, ...].cpu().numpy()}")
        train_metrics['GDL'][epoch-start_epoch, ...] = GDL/numim
        train_metrics['GDLbin'][epoch-start_epoch, ...] = GDLbin/numim

        #validate
        model.eval()
        NN = len(val_loader)
        val_iterator = tqdm(val_loader, ncols=120, total=NN, leave=False)
        
        with torch.no_grad():
            GDL, GDLbin = 0, 0
            epoch_val_metrics = {"Loss": torch.tensor(0.0, device=device), 
                                "Dice": torch.zeros((7,), device=device).type(torch.float32)}
            for data_tuple in val_iterator:
                data, target = [i.to(device) for i in data_tuple[:-1]], data_tuple[-1].to(device)

                out = model(*data)
                target, out = TensorCenterCropping(target, out)
                loss = loss_fn(out, target)
                dice = DicePerClassBinary(out.detach(), target.detach())
                GDL += batchGDL(out.detach(), target.detach()).sum()
                GDLbin += batchGDL(out.detach(), target.detach(), True).sum()
                epoch_val_metrics["Loss"] += loss.item()
                epoch_val_metrics["Dice"] += dice.detach()


                status = {"loss": loss.item(), "Dice": dice.detach().cpu().numpy()}
                val_iterator.set_postfix(status) #description(status)


            #save results
            for i in ['Loss', 'Dice']: #train_metrics:
                val_metrics[f"val_{i}"][epoch-start_epoch, ...] = epoch_val_metrics[i]/NN
            print(f" [VAL] Loss={val_metrics['val_Loss'][epoch-start_epoch]}, GDL={GDL/numimval}, Dice={val_metrics['val_Dice'][epoch-start_epoch, ...].cpu().numpy()}")
            val_metrics['val_GDL'][epoch-start_epoch, ...] = GDL/numimval
            val_metrics['val_GDLbin'][epoch-start_epoch, ...] = GDLbin/numimval


            if best_avg_dice<val_metrics['val_Dice'][epoch-start_epoch, 1:7].mean(): #we have new best epoch
                best_epoch = epoch # - start_epoch
                best_avg_dice = val_metrics['val_Dice'][epoch-start_epoch, 1:7].mean()
                print(f"> Best epoch so far: {best_epoch}")
                #for better eval, let's save the current net, and information about the best epoch:
                with open(f'RESULTS/{args.save_as}_bestepoch.txt', 'a') as f:
                    f.write(str(best_epoch))
                
                torch.save({'epoch': best_epoch,
                            'state_dict': model.state_dict(),
                            'optimizer': optimizer.state_dict()}, f'RESULTS/{args.save_as}_bestepoch')

        #Let's empty cache after each epoch just in case that helps memory issues... 
        #torch.cuda.empty_cache()
        if args.schedule and (epoch+1 % (best_epoch + 10) == 0):  
            for param_group in optimizer.param_groups:
                lr *= 0.5
                param_group['lr'] = lr
                print(f'> New learning Rate: {lr}')

    save_run(train_metrics, val_metrics, model, optimizer, args.save_as, epoch)
    return



def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument("--n_class", type=int, default=7)
    parser.add_argument("--in_channels", nargs='+', type=int, default=[0,1], help="Indxs of channels to use, if not all channels desirable.")
    parser.add_argument("--network", type=str, required=True, help="The network to use")
    parser.add_argument("--extractor_net", type=str, default='resnet34', help="Extractor net name, if using PSPNet.")
    parser.add_argument("--lower_in_channels", nargs='+', type=int, default=[0,1], help="Indxs of in.channels for lower pathway, if using DeepMedic.")
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--n_epoch', nargs='?', type=int, default=100, help='# of the epochs')
    parser.add_argument('--l_rate', nargs='?', type=float, default=5e-4, help='Learning Rate')
    parser.add_argument("--losses", type=str, required=True,
                        help="List of ('loss_name', {'id': list_of_ids}, weight)")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--restore_from", type=str, default='', help="Stored net to restore?")
    parser.add_argument("--save_as", type=str, required=True)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--schedule", action="store_true")
    parser.add_argument("--in3D", action="store_true", help="Use 3D version of nets? Mind that data should be 3D then!")

   # sys.argv = ['Training.py', '--dataset=POEM80_dts', '--batch_size=32', '--network=UNet', '--n_epoch=100', '--l_rate=1e-3',
   #            "--in_channels 0 1 2 4 5", "--lower_in_channels 0 1 2 4 5", "--save_as=unet_dts", '--debug',
   #             "--losses=[('WeightedGeneralizedDice', {'idc': [0.1, 0.5, 0.4, 0.35, 0.5, 0.4, 0.4]}, 1), ('WeightedCrossEntropy', {'idc': [0.1, 1, 1, 1, 1, 1, 1]}, 1)]" ]


    args = parser.parse_args()
    print(args)
    return args
   

if __name__ == '__Training__':
    train(get_args())

train(get_args())

