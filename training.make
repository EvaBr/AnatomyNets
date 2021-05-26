CC = python3.6
SHELL = /usr/bin/zsh

EPC = 100


DATA = --dataset='POEM25_3d'
DATA2 = --dataset='POEM25_dts'

TRN1 = RESULTS/poem/unet RESULTS/poem/deepmed RESULTS/poem/pnet RESULTS/poem/unet3d RESULTS/poem/deepmed3d RESULTS/poem/pnet3d 
TRN2 = RESULTS/poem/unet_dts RESULTS/poem/deepmed_dts RESULTS/poem/pnet_dts RESULTS/poem/unet_dts3d RESULTS/poem/deepmed_dts3d RESULTS/poem/pnet_dts3d

RESULTS/poem/unet: SV = unet
RESULTS/poem/unet: OPT = --losses="[('WeightedGeneralizedDice', {'idc': [0.1, 0.5, 0.4, 0.35, 0.5, 0.4, 0.4]}, 1), \
					('WeightedCrossEntropy', {'idc': [0.1, 1, 1, 1, 1, 1, 1]}, 1)]" \
					"--save_as=$(SV)" '--network=UNet' '--schedule' $(DATA2)


RESULTS/poem/deepmed: SV = deepmed
RESULTS/poem/deepmed: OPT = --losses="[('WeightedGeneralizedDice', {'idc': [0.1, 0.5, 0.4, 0.35, 0.5, 0.4, 0.4]}, 1), \
					('WeightedCrossEntropy', {'idc': [0.1, 1, 1, 1, 1, 1, 1]}, 1)]" \
					"--save_as=$(SV)" '--network=DeepMedic' '--schedule' $(DATA2)


RESULTS/poem/pnet: SV = pnet
RESULTS/poem/pnet: OPT = --losses="[('WeightedGeneralizedDice', {'idc': [0.1, 0.5, 0.4, 0.35, 0.5, 0.4, 0.4]}, 1), \
					('WeightedCrossEntropy', {'idc': [0.1, 1, 1, 1, 1, 1, 1]}, 1)]" \
					"--save_as=$(SV)" '--network=PSPNet' '--schedule' $(DATA2)


RESULTS/poem/unet3d: SV = unet3d
RESULTS/poem/unet3d: OPT = --losses="[('WeightedGeneralizedDice', {'idc': [0.1, 0.5, 0.4, 0.35, 0.5, 0.4, 0.4]}, 1), \
					('WeightedCrossEntropy', {'idc': [0.1, 1, 1, 1, 1, 1, 1]}, 1)]" \
					"--save_as=$(SV)" '--network=UNet' '--schedule' '--in3D' $(DATA)


RESULTS/poem/deepmed3d: SV = deepmed3d
RESULTS/poem/deepmed3d: OPT = --losses="[('WeightedGeneralizedDice', {'idc': [0.1, 0.5, 0.4, 0.35, 0.5, 0.4, 0.4]}, 1), \
					('WeightedCrossEntropy', {'idc': [0.1, 1, 1, 1, 1, 1, 1]}, 1)]" \
					"--save_as=$(SV)" '--network=DeepMedic' '--schedule' '--in3D' $(DATA)


RESULTS/poem/pnet3d: SV = pnet3d
RESULTS/poem/pnet3d: OPT = --losses="[('WeightedGeneralizedDice', {'idc': [0.1, 0.5, 0.4, 0.35, 0.5, 0.4, 0.4]}, 1), \
					('WeightedCrossEntropy', {'idc': [0.1, 1, 1, 1, 1, 1, 1]}, 1)]" \
					"--save_as=$(SV)" '--network=PSPNet' '--schedule' '--in3D' $(DATA)



RESULTS/poem/unet_dts: SV = unet_dts
RESULTS/poem/unet_dts: OPT = --losses="[('WeightedGeneralizedDice', {'idc': [0.1, 0.5, 0.4, 0.35, 0.5, 0.4, 0.4]}, 1), \
					('WeightedCrossEntropy', {'idc': [0.1, 1, 1, 1, 1, 1, 1]}, 1)]" \
					"--save_as=$(SV)" '--network=UNet' '--schedule' $(DATA2)

RESULTS/poem/pnet_dts: SV = pnet_dts
RESULTS/poem/pnet_dts: OPT = --losses="[('WeightedGeneralizedDice', {'idc': [0.1, 0.5, 0.4, 0.35, 0.5, 0.4, 0.4]}, 1), \
					('WeightedCrossEntropy', {'idc': [0.1, 1, 1, 1, 1, 1, 1]}, 1)]" \
					"--save_as=$(SV)" '--network=PSPNet' '--schedule' $(DATA2)

RESULTS/poem/deepmed_dts: SV = deepmed_dts
RESULTS/poem/deepmed_dts: OPT = --losses="[('WeightedGeneralizedDice', {'idc': [0.1, 0.5, 0.4, 0.35, 0.5, 0.4, 0.4]}, 1), \
					('WeightedCrossEntropy', {'idc': [0.1, 1, 1, 1, 1, 1, 1]}, 1)]" \
					"--save_as=$(SV)" '--network=DeepMedic' '--schedule' $(DATA2)



RESULTS/poem/unet_dts3d: SV = unet_dts3d
RESULTS/poem/unet_dts3d: OPT = --losses="[('WeightedGeneralizedDice', {'idc': [0.1, 0.5, 0.4, 0.35, 0.5, 0.4, 0.4]}, 1), \
					('WeightedCrossEntropy', {'idc': [0.1, 1, 1, 1, 1, 1, 1]}, 1)]" \
					"--save_as=$(SV)" '--network=UNet' '--schedule' '--in3D' $(DATA)

RESULTS/poem/pnet_dts3d: SV = pnet_dts3d
RESULTS/poem/pnet_dts3d: OPT = --losses="[('WeightedGeneralizedDice', {'idc': [0.1, 0.5, 0.4, 0.35, 0.5, 0.4, 0.4]}, 1), \
					('WeightedCrossEntropy', {'idc': [0.1, 1, 1, 1, 1, 1, 1]}, 1)]" \
					"--save_as=$(SV)" '--network=PSPNet' '--schedule' '--in3D' $(DATA)

RESULTS/poem/deepmed_dts3d: SV = deepmed_dts3d
RESULTS/poem/deepmed_dts3d: OPT = --losses="[('WeightedGeneralizedDice', {'idc': [0.1, 0.5, 0.4, 0.35, 0.5, 0.4, 0.4]}, 1), \
					('WeightedCrossEntropy', {'idc': [0.1, 1, 1, 1, 1, 1, 1]}, 1)]" \
					"--save_as=$(SV)" '--network=DeepMedic' '--schedule' '--in3D' $(DATA)


all: $(TRN1) $(TRN2)
$(TRN1):
	mkdir -p $@_tmp
	$(CC) $(CFLAGS) Training.py --batch_size=32 --l_rate=1e-3 \
		--n_epoch=$(EPC) --in_channels 0 1 --lower_in_channels 0 1 $(OPT) 
	mv $@_tmp $@ 
	mv RESULTS/$(SV)* $@/.
	

$(TRN2):
	mkdir -p $@_tmp
	$(CC) $(CFLAGS) Training.py --batch_size=32 --l_rate=1e-3 \
		--n_epoch=$(EPC) --in_channels 0 1 2 3 4 --lower_in_channels 0 1 5 $(OPT)
	mv $@_tmp $@ 
	mv RESULTS/$(SV)* $@/.


