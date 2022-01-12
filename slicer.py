import numpy as np 
import torch
from glob import glob
from typing import List, Tuple, Dict, Any

class Slicer():
    def __init__(self, imgsize:Tuple[int], patchsize:int, ch1: List[int], ch2:List[int], in3D:bool, bydim:int=1, step:int=0):
        #identifying parameters:
        self.ch1 = ch1 
        self.ch2 = ch2
        self.in2 = len(ch2)>0 #bool
        self.bydim = bydim
        self.in3D = in3D #bool
        #for use in slicing:
        self.stuckat = (0,0,0)
        self.steps, self.sizes, self.loopends = None, None, None
        if step==0:  #assume nonoverlapping patches
            step = patchsize
        self.getloopends(imgsize, patchsize, step) #sets self.loopends, self.steps
        self.todo = np.prod(self.loopends)

    def getloopends(self, imgsize:Tuple[int], patch:int, step:int): #todo: axial vs coronal slicing in 2D
        #bydim = 1|2, by which dim to cut in 2D
        x, y, z = imgsize
        if self.in3D:
            xend = np.ceil(x/step) #patch) 
            yend = np.ceil(y/step) 
            zend = np.ceil(z/step)
            self.sizes = (patch, patch, patch)
            self.steps = (step, step, step)
        elif self.bydim==1:
            xend = np.ceil(x/step) 
            yend = y 
            zend = np.ceil(z/step)
            self.sizes = (patch, 1, patch)
            self.steps = (step, 1, step)
        elif self.bydim==2:
            xend = np.ceil(x/step)
            yend = np.ceil(y/step) 
            zend = z 
            self.sizes = (patch, patch, 1)
            self.steps = (step, step, 1)
        else:
            raise "Not supported!"
        self.loopends = (xend, yend, zend)

    def next(self) -> Tuple[int]:
        x0, y0, z0 = self.stuckat
        xs, ys, zs = self.steps
        xe, ye, ze = self.loopends 
        xp, yp, zp = self.sizes

        #newslice = f"{x0*xs}:{(x0+1)*xs}, {y0*ys}:{(y0+1)*ys}, {z0*zs}:{(z0+1)*zs}"
        newslice = (x0*xs, x0*xs + xp, y0*ys, y0*ys + yp, z0*zs, z0*zs + zp)

        if self.todo==0: #we're done now.
            assert x0==y0==z0==0, (x0,y0,z0) #sanity check
            return None
       
        #update how far we've come now, after taking next slice:
        x0 = x0+1 #try going a step further in x direction first.
        if x0==xe: #finished that row, but as the next one starts, we need to start form 0 again..
            x0 = 0
            y0 = y0+1    
            if y0==ye: #finished that row, but as the next one starts, we need to start form 0 again..
                y0 = 0
                z0 = z0+1
                if z0==ze: #finished that row, but as the next one starts, we need to start form 0 again..
                    z0 = 0

        self.stuckat = (x0, y0, z0) #this will not get back to (0,0,0), sice we return None before.
        self.todo = self.todo-1
        return newslice
    
    def back_to_start(self):
        self.todo = np.prod(self.loopends)
        self.stuckat = (0, 0, 0)
    
    def get_batch(self, batchsize:int) -> Tuple[List[str], List[str], List[str]]:
        batch = [self.next() for i in range(batchsize)] 
        base = [self.slice_fix(b) for b in batch if b!=None] #IGNORE NONE in the batch
        batchgt = [":," + bgt for bgt in base]
        batchin1 = [f"{self.ch1}," + bgt for bgt in base]
        batchin2 = []
        if self.in2:
            batchin1 = [f"{self.ch1}," + self.slice_fix(b, ch=1, deepmed=True) for b in batch if b!=None]
            batchin2 = [f"{self.ch2}," + self.slice_fix(b, ch=2, deepmed=True) for b in batch if b!=None]
        return batchgt, batchin1, batchin2
        

    def slice_fix(self, slajs:Tuple[int], ch:int = 0, deepmed:bool = False) -> str:
        """old args: add:int = 0, shift:int = 0
            with add=N and shift=K, returns slice with added size of N pixels on all sides, 
            with its center shifted by K to the right. ADD = 0(gt and non deepmed ins),
            8 (deepmed in1) or 16 (deepmed in2). SHIFT = 0 (non deepmed or in2 deepmed), 
            8 (deepmed in1) or 16 (deepmed gt).
            instead: deepmed = True/False, ch = 0(gt)/1(in1)/2(in2)"""
        adds = {True: [0, 16, 32+16], False: [0, 0, 0]} 
        #gt is not padded before slicing, so no shift necessary even with deepmed!
        shifts = {True: [0, 16, 0], False: [0, 0, 0]}
        add = adds[deepmed][ch]
        shift = shifts[deepmed][ch]

        downstr = [""]*3
        if deepmed and ch==2: #doing deepmed downsampled input
            downstr = [":3"]*3 
        adding = [shift,add+shift, shift,add+shift, shift,add+shift]
        if not self.in3D:
            adding[2*self.bydim:2*self.bydim+2] = [0,0]
            downstr[self.bydim] = ""
        
        newslajs = (a+b for a,b in zip(slajs, adding))

        x,xx,y,yy,z,zz = newslajs
            
        return f"{x}:{xx}{downstr[0]}, {y}:{yy}{downstr[1]}, {z}:{zz}{downstr[2]}"
        
