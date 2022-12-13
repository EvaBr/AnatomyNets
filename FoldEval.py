import torch 
from pathlib import Path
import re
import argparse
from Postprocessing import Compute3DDice


def testFolds(netname="VanillaCNN/rep1/van_1f/vanilla",patchsize = 25, best='_80'):
    #hardcoded_: for it to work, always use .../repN/net_1f/.... It expects 1f! 
    #   (goes through all folds and dt/no dt automatically)
    test = {'1': [500051, 500022, 500379, 500167, 500327, 500117, 500297, 500077, 500280, 500253], 
        '2': [500433, 500179, 500159, 500158, 500487, 500268, 500086, 500075, 500281, 500056], 
        '3': [500235, 500242, 500017, 500204, 500358, 500357, 500348, 500346, 500304, 500473], 
        '4': [500429, 500018, 500062, 500403, 500061, 500347, 500354, 500316, 500321, 500288], 
        '5': [500291, 500424, 500406, 500053, 500026, 500241, 500205, 500395, 500318, 500124]}


    pot = Path(netname).name
    
    netabbrev='van' if pot[0]=='v' else ('un' if pot[0]=='u' else 'dm')
    resultFolder = f"RESULTS/OUTS/{netabbrev}-5fold/" #hardcoded to my dir structure for now
    Path(resultFolder).mkdir(parents=True, exist_ok=True)

    splitname = netname.split("_1f")
    rep = re.findall(r"rep[0-5]", netname)[0]
    assert len(splitname)==2, splitname

    for fold in test:
        torch.cuda.empty_cache()
        name = splitname[0] + f"_{fold}f_dt" + splitname[1] #f"/van_{fold}f/vanilla"
        print(name)
        a = Compute3DDice(test[fold], name, patchsize, 
              best=best, step=10, saveout=False, savename=resultFolder+f'dices_{netabbrev}_5fold_dt_{rep}')
        torch.cuda.empty_cache()
        #print(a)

        name = splitname[0] + f"_{fold}f" + splitname[1] #f"/van_{fold}f/vanilla"
        print(name)
        torch.cuda.empty_cache()
        a = Compute3DDice(test[fold], name, patchsize, 
                best=best, step=10, saveout=False, savename=resultFolder+f'dices_{netabbrev}_5fold_{rep}')
        torch.cuda.empty_cache()
        #print(a)
    return 


def getargs():
    parser = argparse.ArgumentParser(description='Validation Eval')
    parser.add_argument("--path", required=True, type=str)
    parser.add_argument("--patchsize", type=int, default=25)
    parser.add_argument("--best", type=str, default="_80")

    args = parser.parse_args()
    print(args)
    return args


if __name__ == '__main__':
    args = getargs()
    testFolds(args.path, args.patchsize, args.best)
