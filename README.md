# AnatomyNets
AnatomyAware Nets v2

Currently just a collection of three segmentation networks (U-Net [1], PSP-Net [2] (with resnet extractors [3]), DeepMedic [4]), that allow for three different inclusions of wider context (skip connections, pyramid pooling - dilated convolutions, downsampled pathway), useful particularly when training on smaller patches.

To run the training with any of these networks on your own data, simply comment out `sys.argvs` line in *Training.py* (useful for debugging), and run 
```python3 Training.py --flags``` 
in the terminal, together with desired flags and parameters (see argument parser in Training.py for more info). After training is finished, the results will be saved in folder RESULTS: a file with trained pytorch network checkpoint, a txt file with all set parameters and a csv file with all metrics info per epoch. 

Your data should be structured in the following way: 

 
    ├── ...
    ├── Training.py             # Main script for training    
    ├── DATAFOLDER              # Main Data folder. Make one for each individual dataset.
    │   ├── TRAIN               # Folder for training data
    │   |   ├── in1                # main input images/patches, in npy format
    │   |   ├── in2                # corresponding downsampled input images/patches, in npy format (only used for DeepMedic)
    │   |   └── gt                 # correspoinding ground truth images/patches
    │   ├── VAL                 # Folder for validation data
    │       ├── in1                # main input images/patches, in npy format
    │       ├── in2                # corresponding downsampled input images/patches, in npy format (only used for DeepMedic)
    │       └── gt                 # correspoinding ground truth images/patches
    └── ...
    
    
The script *Slicing.py* contains (hardcoded to my own dataset) code for slicing patches and saving them as *.npy* under an appropriate structure.
The script *Postprocessing.py* contains (again somewhat hardcoded to my own data) functions for ad-hoc visualizing and comparing training curves and segmentation output. 


[1] U-Net paper: https://arxiv.org/pdf/1505.04597.pdf </br>
[2] PSP-Net paper: https://arxiv.org/pdf/1612.01105.pdf</br>
[3] extractors implementation shamelesly taken from: https://github.com/Lextal/pspnet-pytorch/blob/master/extractors.py</br>
[4] DeepMedic paper: https://www.sciencedirect.com/science/article/pii/S1361841516301839
