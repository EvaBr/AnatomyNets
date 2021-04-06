from torch import nn, einsum, Tensor
import torch

class MultiTaskLoss():
    def __init__(self, lossdict, trainable=False):
        losses = eval(lossdict)
        loss_fns: List[Callable] = []
        loss_ws: List[float] = []
        for loss_name, loss_params, weight in losses:
            loss_fns.append(globals()[loss_name](**loss_params))
            loss_ws.append(weight)

        self.losses = loss_fns
        self.sigma = torch.tensor(loss_ws)
        if trainable:
            self.sigma = nn.Parameter(torch.tensor(loss_ws))
        
    def __call__(self, out, target):
        loss = torch.as_tensor([s*loss_fn(out, target) for s, loss_fn in zip(self.sigma, self.losses)])
        loss.requires_grad_()
        return loss.sum()


class CrossEntropy():
    def __init__(self, **kwargs):
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]

    def __call__(self, probs: Tensor, target: Tensor) -> Tensor:
        log_p: Tensor = probs[:, self.idc, ...] #+ 1e-10).log()
        mask: Tensor = target[:, self.idc, ...].type(torch.float32)

        loss = - einsum("bcwh,bcwh->", mask, log_p)
        loss /= max(mask.sum(), 1e-10) #mask.sum() + 1e-10
        return loss


class GeneralizedDice():
    def __init__(self, **kwargs):
        self.idc: List[int] = kwargs["idc"]
        self.epsilon: float = kwargs["epsilon"] if "epsilon" in kwargs else 1.
        self.strategy: float = kwargs["strategy"] if "strategy" in kwargs else None
        assert self.strategy in [None, "volume", "normalize"], "Wrong option when choosing strategy."


    def __call__(self, probs: Tensor, target: Tensor) -> Tensor:
        pc = probs[:, self.idc, ...].type(torch.float32).exp()
        tc = target[:, self.idc, ...].type(torch.float32)

        #w: Tensor = 1. / ((einsum("bcwh->bc", tc).type(torch.float32) + 1e-10) ** 2)
        w: Tensor = einsum("bcwh->bc", tc).float()
        if self.strategy=="volume":
            w = torch.where(w!=0., 
                1/torch.max(w, torch.ones_like(w)*1e-4)**2,
                torch.zeros_like(w)
            )
        elif self.strategy=="normalize":
            w = torch.div(w.T, w.sum(1)).T
            
        elif self.strategy==None:
            w = 1. / ( w + 1e-10 )**2

        intersection: Tensor = w * einsum("bcwh,bcwh->bc", pc, tc)
        union: Tensor = w * (einsum("bcwh->bc", pc) + einsum("bcwh->bc", tc))

        divided: Tensor = 1 - (2 * einsum("bc->b", intersection) + self.epsilon) / (einsum("bc->b", union) + self.epsilon)

        loss = divided.mean()
        return loss


def DicePerClass(probs: Tensor, target: Tensor):
    pc = probs.type(torch.float32).exp()
    tc = target.type(torch.float32)
    
    w: Tensor = 1 / ((einsum("bcwh->bc", tc).type(torch.float32) + 1e-10) ** 2)
    intersection: Tensor = w * einsum("bcwh,bcwh->bc", pc, tc)
    union: Tensor = w * (einsum("bcwh->bc", pc) + einsum("bcwh->bc", tc))

    divided: Tensor = torch.clamp(2 * (intersection + 1e-10) / (union + 1e-10), max=1.) 
    #occasionally, if class neither in GT nor OUT in the whole batch, the output might be 2 -.- => clamping needed. (IS IT ENOUGH?)

    loss = divided.mean(dim=0) #average over batch. Output size should be C=nb_classes
    return loss