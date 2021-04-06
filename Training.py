import Networks
from Losses import MultiTaskLoss, DicePerClass, GeneralizedDice, CrossEntropy
from Dataloaders import get_loaders
import argparse
import torch
from torch import optim, nn
from tqdm import tqdm
import numpy as np
import pandas as pd
import sys
import numpy as np
import os
np.set_printoptions(precision=3, suppress=True)
torch.set_printoptions(precision=3, sci_mode=False)

def CenterCropTensor(tgt, x):
    xs2, xs3 = x.shape[2], x.shape[3]
    tg2, tg3 = tgt.shape[2], tgt.shape[3]
    diffY = abs(xs2 - tg2)//2
    diffX = abs(xs3 - tg3)//2
    ostanek = abs(xs2-tg2)%2
    
    if xs2>tg2: 
        x = x[..., diffX:xs2-diffX-ostanek, diffY:xs3-diffY-ostanek]
    else: 
        tgt = tgt[..., diffX:tg2-diffX-ostanek, diffY:tg3-diffY-ostanek] 
    return tgt, x

def save_run(train_m, val_m, model, optimizer, save_to, epoch):
    """Saves the final state of network for re-loading, and CSV of metrics, and a txt file of args used for the run."""

    if not os.path.exists('RESULTS'):
        os.mkdir('RESULTS')
    joint_dic = {**train_m, **val_m}
    df = pd.DataFrame({k: v.cpu().numpy().tolist() for k,v in joint_dic.items()})
    cols = ['Dice_bck','Dice_Bladder', 'Dice_KidneyL', 'Dice_Liver', 'Dice_Pancreas', 'Dice_Spleen', 'Dice_KidneyR']
    cols_v = [f"val_{c}" for c in cols]
    df = pd.concat([df['Loss'], pd.DataFrame(df['Dice'].values.tolist(), columns=cols), 
                df['val_Loss'], pd.DataFrame(df['val_Dice'].values.tolist(), columns=cols_v)], axis=1)
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

    net = getattr(Networks, args.network)(args.in_channels, args.n_class, args.lower_in_channels, args.extractor_net)
    net = net.float()
    to_init = Networks.weights_init(args.network)
    net.apply(to_init)

    start_epoch = 0
    optimizer =  optim.Adam(net.parameters(), lr=args.l_rate)
    if args.restore_from!="":
        #restore pretrained and saved network:
        loaded = torch.load(args.restore_from)
        #assert loaded['name'] == args.network
        net.load_state_dict(loaded['state_dict'])
        optimizer.load_state_dict(loaded['optimizer'])
        start_epoch = loaded['epoch']

        # now individually transfer the optimizer parts...
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
    
    model = net.to(device)
    loss = MultiTaskLoss(args.losses)
    #loss = GeneralizedDice(**{'idc': [3]})
    #loss = CrossEntropy(**{'idc': [0]})
    #loss = GeneralizedDice(**{'idc': [0,1,2,3,4,5,6]})
    train_loader, val_loader = get_loaders(args.network, args.dataset, args.n_class, args.batch_size, args.debug)

    return model, optimizer, loss, train_loader, val_loader, device, start_epoch



def train(args: argparse.Namespace):
    model, optimizer, loss_fn, train_loader, val_loader, device, start_epoch = setup(args)

    train_metrics = {"Loss": torch.zeros((args.n_epoch, ), device=device).type(torch.float32), 
                     "Dice": torch.zeros((args.n_epoch, args.n_class), device=device).type(torch.float32)}
    val_metrics = {"val_Loss": torch.zeros((args.n_epoch, ), device=device).type(torch.float32), 
                   "val_Dice": torch.zeros((args.n_epoch, args.n_class), device=device).type(torch.float32)}
    for epoch in range(start_epoch, args.n_epoch+start_epoch):
        #train
        model.train()
        NN = len(train_loader)
        train_iterator = tqdm(train_loader, ncols=150, total=NN, leave=False)
        
        epoch_train_metrics = {"Loss": torch.tensor(0.0, device=device), 
                            "Dice": torch.zeros((7,), device=device).type(torch.float32)}
        for data_tuple in train_iterator:
            optimizer.zero_grad()
            data, target = [i.to(device) for i in data_tuple[:-1]], data_tuple[-1].to(device)
 #           print([i.shape for i in data])
            out = model(*data)
#            print(f"target: {target.shape}, out: {out.shape}")
            #for some nets, output will be smaller than target. Crop target centrally to match output:
 #           print(f"data0: {data[0].shape}, target: {target.shape}, out: {out.shape}")
            target, out = CenterCropTensor(target, out)
 #           print(f"new target: {target.shape}, new out: {out.shape}")
            loss = loss_fn(out, target)

            loss.backward()
            optimizer.step()

            dice = DicePerClass(out, target).detach()
            epoch_train_metrics["Loss"] += loss.detach()
            epoch_train_metrics["Dice"] += dice.detach()

            status = {"loss": loss.item(), "Dice": dice.detach().cpu().numpy()} #is there a way to print nicely without copying to cpu?
            train_iterator.set_postfix(status) #description(status)
        
        #save results:
        for i in train_metrics:
            train_metrics[i][epoch-start_epoch, ...] = epoch_train_metrics[i]/NN
        print(f"EPOCH {epoch}: \n [TRAIN] Loss={train_metrics['Loss'][epoch-start_epoch]}, Dice={train_metrics['Dice'][epoch-start_epoch, ...].cpu().numpy()}")
        
        #validate
        model.eval()
        NN = len(val_loader)
        val_iterator = tqdm(val_loader, ncols=150, total=NN, leave=False)

        epoch_val_metrics = {"Loss": torch.tensor(0.0, device=device), 
                            "Dice": torch.zeros((7,), device=device).type(torch.float32)}
        for data_tuple in val_iterator:
            data, target = [i.to(device) for i in data_tuple[:-1]], data_tuple[-1].to(device)

            out = model(*data)
            target, out = CenterCropTensor(target, out)
            loss = loss_fn(out, target)
            dice = DicePerClass(out, target)
            epoch_val_metrics["Loss"] += loss.detach()
            epoch_val_metrics["Dice"] += dice.detach()


            status = {"loss": loss.item(), "Dice":dice.detach().cpu().numpy()}
            val_iterator.set_postfix(status) #description(status)


        #save results
        for i in train_metrics:
            val_metrics[f"val_{i}"][epoch-start_epoch, ...] = epoch_val_metrics[i]/NN
        print(f" [VAL] Loss={val_metrics['val_Loss'][epoch-start_epoch]}, Dice={val_metrics['val_Dice'][epoch-start_epoch, ...].cpu().numpy()}")

    
    save_run(train_metrics, val_metrics, model, optimizer, args.save_as, epoch)






def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument("--n_class", type=int, default=7)
    parser.add_argument("--in_channels", type=int, default=2)
    parser.add_argument("--network", type=str, required=True, help="The network to use")
    parser.add_argument("--extractor_net", type=str, default='resnet34', help="Extractor net name, if using PSPNet.")
    parser.add_argument("--lower_in_channels", type=int, default=2, help="Nr of in.channels for lower pathway, if using DeepMedic.")
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--n_epoch', nargs='?', type=int, default=100, help='# of the epochs')
    parser.add_argument('--l_rate', nargs='?', type=float, default=5e-4, help='Learning Rate')
    parser.add_argument("--losses", type=str, required=True,
                        help="List of ('loss_name', {'id': list_of_ids}, weight)")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--restore_from", type=str, default='', help="Stored net to restore?")
    parser.add_argument("--save_as", type=str, required=True)
    parser.add_argument("--cpu", action="store_true")

    sys.argv = ['Training.py', '--dataset=POEM', '--batch_size=32', "--network=UNet", "--n_epoch=100", "--l_rate=5e-2",
                "--losses=[('GeneralizedDice', {'idc': [1,2,4,5,6], 'epsilon': 1e-10}, 1)]" , 
                "--save_as=unet", "--debug"]

    args = parser.parse_args()
    print(args)
    return args
   

if __name__ == '__Training__':
    train(get_args())

train(get_args())
