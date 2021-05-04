CC = python3.6
SHELL = /usr/bin/zsh

EPC = 50


DATA = --dataset='POEM'

TRN = RESULTS/poem/gdl1 RESULTS/poem/gdl1_w RESULTS/poem/gdl2 RESULTS/poem/gdl2_w 


RESULTS/poem/gdl1: SV = unet_gdl
RESULTS/poem/gdl1: OPT = --losses="[('GeneralizedDice', {'idc': [1, 4]}, 0.5), ('GeneralizedDice', {'idc': [2, 5, 6]}, 0.4), \
					('GeneralizedDice', {'idc': [3]},  0.35), ('CrossEntropy', {'idc': [0]}, 0.15), \
					('CrossEntropy', {'idc': [1,2,3,4,5,6]}, 1)]" \
					"--save_as=$(SV)" '--network=UNet' '--schedule'


RESULTS/poem/gdl1_w: SV = unet_gdl_w
RESULTS/poem/gdl1_w: OPT = --losses="[('WeightedGeneralizedDice', {'idc': [0.15, 0.5, 0.4, 0.35, 0.5, 0.4, 0.4]}, 1), \
					('WeightedCrossEntropy', {'idc': [0.15, 1, 1, 1, 1, 1, 1]}, 1)]" \
					"--save_as=$(SV)" '--network=UNet' '--schedule'

RESULTS/poem/gdl2: SV = unet2_gdl
RESULTS/poem/gdl2: OPT = --losses="[('GeneralizedDice', {'idc': [1, 4]}, 0.5), ('GeneralizedDice', {'idc': [2, 5, 6]}, 0.4), \
					('GeneralizedDice', {'idc': [3]},  0.35), ('CrossEntropy', {'idc': [0]}, 0.15), \
					('CrossEntropy', {'idc': [1,2,3,4,5,6]}, 1)]" \
					"--save_as=$(SV)" '--network=UNet2' '--schedule'


RESULTS/poem/gdl2_w: SV = unet2_gdl_w
RESULTS/poem/gdl2_w: OPT = --losses="[('WeightedGeneralizedDice', {'idc': [0.15, 0.5, 0.4, 0.35, 0.5, 0.4, 0.4]}, 1), \
					('WeightedCrossEntropy', {'idc': [0.15, 1, 1, 1, 1, 1, 1]}, 1)]" \
					"--save_as=$(SV)" '--network=UNet2' '--schedule'

all: $(TRN)
$(TRN):
	mkdir -p $@_tmp
	$(CC) $(CFLAGS) Training.py --batch_size=8 --l_rate=1e-3 \
		--n_epoch=$(EPC) $(OPT) $(DATA)
	mv $@_tmp $@ 
	mv RESULTS/$(SV)* $@/.
	