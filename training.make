CC = python3.6
SHELL = /usr/bin/zsh

EPC = 50


DATA = --dataset='POEM80_dt'

TRN1 = RESULTS/poem/unet_w RESULTS/poem/deepmed_w RESULTS/poem/pnet_w 
TRN2 = RESULTS/poem/unet_w_dt RESULTS/poem/deepmed_w_dt RESULTS/poem/pnet_w_dt

RESULTS/poem/unet_w: SV = unet
RESULTS/poem/unet_w: OPT = --losses="[('WeightedGeneralizedDice', {'idc': [0.1, 0.5, 0.4, 0.35, 0.5, 0.4, 0.4]}, 1), \
					('WeightedCrossEntropy', {'idc': [0.1, 1, 1, 1, 1, 1, 1]}, 1)]" \
					"--save_as=$(SV)" '--network=UNet' '--schedule'


RESULTS/poem/deepmed_w: SV = deepmed
RESULTS/poem/deepmed_w: OPT = --losses="[('WeightedGeneralizedDice', {'idc': [0.1, 0.5, 0.4, 0.35, 0.5, 0.4, 0.4]}, 1), \
					('WeightedCrossEntropy', {'idc': [0.1, 1, 1, 1, 1, 1, 1]}, 1)]" \
					"--save_as=$(SV)" '--network=DeepMedic' '--schedule'


RESULTS/poem/pnet_w: SV = pnet
RESULTS/poem/pnet_w: OPT = --losses="[('WeightedGeneralizedDice', {'idc': [0.1, 0.5, 0.4, 0.35, 0.5, 0.4, 0.4]}, 1), \
					('WeightedCrossEntropy', {'idc': [0.1, 1, 1, 1, 1, 1, 1]}, 1)]" \
					"--save_as=$(SV)" '--network=PSPNet' '--schedule'


RESULTS/poem/unet_w_dt: SV = unet_dt
RESULTS/poem/unet_w_dt: OPT = --losses="[('WeightedGeneralizedDice', {'idc': [0.1, 0.5, 0.4, 0.35, 0.5, 0.4, 0.4]}, 1), \
					('WeightedCrossEntropy', {'idc': [0.1, 1, 1, 1, 1, 1, 1]}, 1)]" \
					"--save_as=$(SV)" '--network=UNet' '--schedule'


RESULTS/poem/deepmed_w_dt: SV = deepmed_dt
RESULTS/poem/deepmed_w_dt: OPT = --losses="[('WeightedGeneralizedDice', {'idc': [0.1, 0.5, 0.4, 0.35, 0.5, 0.4, 0.4]}, 1), \
					('WeightedCrossEntropy', {'idc': [0.1, 1, 1, 1, 1, 1, 1]}, 1)]" \
					"--save_as=$(SV)" '--network=DeepMedic' '--schedule'


RESULTS/poem/pnet_w_dt: SV = pnet_dt
RESULTS/poem/pnet_w_dt: OPT = --losses="[('WeightedGeneralizedDice', {'idc': [0.1, 0.5, 0.4, 0.35, 0.5, 0.4, 0.4]}, 1), \
					('WeightedCrossEntropy', {'idc': [0.1, 1, 1, 1, 1, 1, 1]}, 1)]" \
					"--save_as=$(SV)" '--network=PSPNet' '--schedule'

all: $(TRN1) $(TRN2)
$(TRN1):
	mkdir -p $@_tmp
	$(CC) $(CFLAGS) Training.py --batch_size=8 --l_rate=1e-3 \
		--n_epoch=$(EPC) --in_chan 0 1 --lower_in_chan 0 1 $(OPT) $(DATA)
	mv $@_tmp $@ 
	mv RESULTS/$(SV)* $@/.
	
$(TRN2):
	mkdir -p $@_tmp
	$(CC) $(CFLAGS) Training.py --batch_size=8 --l_rate=1e-3 \
		--n_epoch=$(EPC) --in_chan 0 1 2 3 --lower_in_chan 0 1 2 3 $(OPT) $(DATA)
	mv $@_tmp $@ 
	mv RESULTS/$(SV)* $@/.
