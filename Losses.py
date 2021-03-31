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
        # Self.idc is used to filter out some classes of the target mask. Use fancy indexing
        self.idc: List[int] = kwargs["idc"]

    def __call__(self, probs: Tensor, target: Tensor) -> Tensor:
        pc = probs[:, self.idc, ...].type(torch.float32).exp()
        tc = target[:, self.idc, ...].type(torch.float32)

        w: Tensor = 1 / ((einsum("bcwh->bc", tc).type(torch.float32) + 1e-10) ** 2)
        intersection: Tensor = w * einsum("bcwh,bcwh->bc", pc, tc)
        union: Tensor = w * (einsum("bcwh->bc", pc) + einsum("bcwh->bc", tc))

        divided: Tensor = 1 - 2 * (einsum("bc->b", intersection) + 1e-10) / (einsum("bc->b", union) + 1e-10)

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