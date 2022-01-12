from torch import nn, einsum, Tensor
import torch
from functools import reduce
from operator import add
from typing import List, cast

class MultiTaskLoss():
    def __init__(self, lossdict, device, in3d=False, trainable=False):
        losses = eval(lossdict)
        loss_fns: List[Callable] = []
        loss_ws: List[float] = []
        for loss_name, loss_params, weight in losses:
            loss_params['device'] = device
          #  loss_params['in3d'] = in3d
            loss_fns.append(globals()[loss_name](**loss_params))
            loss_ws.append(weight)

        self.losses = loss_fns
        self.sigma = torch.tensor(loss_ws)
        if trainable:
            self.sigma = nn.Parameter(torch.tensor(loss_ws))
        
    def __call__(self, out: Tensor, target: Tensor):
        losses: List[Tensor] = [s*loss_fn(out, target) for s, loss_fn in zip(self.sigma, self.losses)]
       # assert all(l.requires_grad for l in losses)
        return reduce(add, losses)


class CrossEntropy():
    def __init__(self, **kwargs):
        self.idc: List[int] = kwargs["idc"]

    def __call__(self, probs: Tensor, target: Tensor) -> Tensor:
        log_p: Tensor = probs[:, self.idc, ...] #+ 1e-10).log()
        mask: Tensor = target[:, self.idc, ...].type(torch.float32)

        loss = - einsum("bchw...,bchw...->", mask, log_p)
        loss /= max(mask.sum(), 1e-10) #mask.sum() + 1e-10
        return loss

class WeightedCrossEntropy():
    def __init__(self, **kwargs):
        idc = kwargs["idc"] #now these are weights that we apply. 
        self.idc: List[int] = [i for i,v in enumerate(idc) if v>0]
        #If a class should be ignored, simply set weight=0 for that class.
        device = kwargs["device"]
        self.weights = torch.tensor([i for i in idc if i>0]).float().to(device)
        #self.is3d = kwargs["in3d"]
        #self.str = "bcwhd" if self.is3d else "bcwh"


    def __call__(self, probs: Tensor, target: Tensor) -> Tensor:
        log_p: Tensor = probs[:, self.idc, ...] #+ 1e-10).log()
        mask: Tensor = target[:, self.idc, ...].type(torch.float32)

        loss = - einsum("bcwh...,bcwh...->c", mask, log_p)
        loss = torch.dot(loss, self.weights)

        mask = einsum("bcwh...->c", mask)
        mask = torch.dot(mask, self.weights)

        #loss /= max(mask.sum(), 1e-10) #mask.sum() + 1e-10
        loss /= mask + 1e-10
        return loss

class GeneralizedDice():
    def __init__(self, **kwargs):
        self.idc: List[int] = kwargs["idc"]
        self.epsilon: float = kwargs["epsilon"] if "epsilon" in kwargs else 1
        self.strategy: float = kwargs["strategy"] if "strategy" in kwargs else None
        assert self.strategy in [None, "volume", "normalize"], "Wrong option when choosing strategy."


    def __call__(self, probs: Tensor, target: Tensor) -> Tensor:
        pc = probs[:, self.idc, ...].exp().type(torch.float32)
        tc = target[:, self.idc, ...].type(torch.float32)

        w: Tensor = 1. / ((einsum("bc...->bc", tc).type(torch.float32) + 1e-10) ** 2)
        #w: Tensor = einsum("bcwh->bc", tc).float()
        #if self.strategy=="volume":
        #    w = torch.where(w!=0., 
        #        1/torch.max(w, torch.ones_like(w)*1e-4)**2,
        #        torch.zeros_like(w)
        #    )
        #elif self.strategy=="normalize":
        #    w = torch.div(w.T, w.sum(1)).T
        #elif self.strategy==None:
        #    w = 1. / (( w + 1e-10 )**2)

        intersection: Tensor = w * einsum("bc...,bc...->bc", pc, tc)
        union: Tensor = w * (einsum("bc...->bc", pc) + einsum("bc...->bc", tc))

        divided: Tensor = 1 - (2 * einsum("bc->b", intersection) + self.epsilon) / (einsum("bc->b", union) + self.epsilon)

        loss = divided.mean()
        return loss


class WeightedGeneralizedDice():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        idc = kwargs["idc"] #now these are weights that we apply. 
        self.idc: List[int] = [i for i,v in enumerate(idc) if v>0]
        #If a class should be ignored, simply set weight=0 for that class.
        device = kwargs["device"]
        self.weights = torch.tensor([i for i in idc if i>0]).float().to(device)


    def __call__(self, probs: Tensor, target: Tensor) -> Tensor:
        pc = probs[:, self.idc, ...].exp().type(torch.float32)
        tc = target[:, self.idc, ...].type(torch.float32)

        #OPTION 1: instead of dynamically changing weights batch-based, keep them static based on input weights
        w: Tensor = 1 / ((self.weights+1e-10)**2)
    
        intersection: Tensor = w * einsum("bkwh...,bkwh...->bk", pc, tc)
        union: Tensor = w * (einsum("bkwh...->bk", pc) + einsum("bkwh...->bk", tc))

        divided: Tensor = 1 - (2 * einsum("bk->b", intersection) + 1e-10) / (einsum("bk->b", union) + 1e-10)

        #OPTION 2: imitate the computation that happens if you put in multiple/per-class GDL losses as args 
    #    w: Tensor = 1 / ((einsum("bkwh->bk", tc).type(torch.float32) + 1e-10) ** 2)
    #    intersection: Tensor = w * einsum("bkwh,bkwh->bk", pc, tc)
    #    union: Tensor = w * (einsum("bkwh->bk", pc) + einsum("bkwh->bk", tc))

    #    divided: Tensor = self.weights.sum() - 2 * einsum("bk->b", (intersection + 1e-10) / (union + 1e-10) * self.weights)

        loss = divided.mean()
        return loss

def DicePerClass(probs: Tensor, target: Tensor):
    pc = probs.type(torch.float32).exp()
    tc = target.type(torch.float32)
    
    intersection: Tensor = einsum("bcwh...,bcwh...->bc", pc, tc)
    union: Tensor = (einsum("bcwh...->bc", pc) + einsum("bcwh...->bc", tc))

    divided: Tensor = (2 * intersection + 1e-10) / (union + 1e-10)
    #if class present neither in GT nor OUT in the whole batch, the output==1

    loss = divided.mean(dim=0) #average over batch. Output size should be C=nb_classes
    return loss


def AllDices(probs: Tensor, target: Tensor):
    pc = probs.type(torch.float32).exp()
    pc_bin = torch.argmax(pc, dim=1, keepdim=True) #better report it on binary outs.
    pc = pc.zero_()
    pc = pc.scatter_(1, pc_bin, 1) #torch.ones(pc_bin.size()))
    tc = target.type(torch.float32)
    
    intersection: Tensor = einsum("bcwh...,bcwh...->bc", pc, tc)
    union: Tensor = (einsum("bcwh...->bc", pc) + einsum("bcwh...->bc", tc))

    divided: Tensor = (2 * intersection + 1e-10) / (union + 1e-10)
    #if class present neither in GT nor OUT in the whole batch, the output==1

    return divided

def DicePerClassBinary(probs: Tensor, target: Tensor):
    divided = AllDices(probs, target)
    return divided.mean(dim=0)

def batchGDL(probs: Tensor, target: Tensor, binary: bool = False):
    pc = probs.type(torch.float32).exp()
    if binary:
        pc_bin = torch.argmax(pc, dim=1, keepdim=True) #better report it on binary outs.
        pc = pc.zero_()
        pc = pc.scatter_(1, pc_bin, 1)
    tc = target.type(torch.float32)

    intersection: Tensor = einsum("bcwh...,bcwh...->b", pc, tc)
    union: Tensor = einsum("bcwh...->b", pc) + einsum("bcwh...->b", tc) #ne bo nikoli 0. zmeri bo == numel. 

    divided: Tensor = 2*intersection / union
   
    return divided

def subjectDices(probs: Tensor, target: Tensor, binary: bool = False):
    pc = probs.type(torch.float32).exp()
    if binary:
        pc_bin = torch.argmax(pc, dim=1, keepdim=True) #better report it on binary outs.
        pc = pc.zero_()
        pc = pc.scatter_(1, pc_bin, 1)
    tc = target.type(torch.float32)

    intersection: Tensor = einsum("bcwh...,bcwh...->c", pc, tc)
    union: Tensor = einsum("bcwh...->c", pc) + einsum("bcwh...->c", tc) #ne bo nikoli 0. zmeri bo vsaj == numel.Ker vsi klasi obstajajo. 

    divided: Tensor = 2*intersection / union
    GDL: Tensor = 2*intersection.sum() / union.sum()
   
    return divided, GDL

class FocalLoss():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]
        self.gamma: float = kwargs["gamma"]
        print(f"Initialized {self.__class__.__name__} with {kwargs}")

    def __call__(self, probs: Tensor, target: Tensor) -> Tensor:

        masked_probs: Tensor = probs[:, self.idc, ...]
        log_p: Tensor = (masked_probs + 1e-10).log()
        mask: Tensor = cast(Tensor, target[:, self.idc, ...].type(torch.float32))

        w: Tensor = (1 - masked_probs)**self.gamma
        loss = - einsum("bk...,bk...,bk...->", w, mask, log_p)
        loss /= mask.sum() + 1e-10

        return loss
