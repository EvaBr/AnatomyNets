CC = python3.6
SHELL = /usr/bin/zsh

EPC = 100


DATA = --dataset='POEM25_3D'
DATA1 = --dataset='POEM25'
DATA2 = --dataset='POEM25_2'

TRN1 = RESULTS/poem/unet3d RESULTS/poem/deepmed3d RESULTS/poem/pnet3d 
TRN2 = RESULTS/poem/unet1 RESULTS/poem/deepmed1 RESULTS/poem/pnet1 RESULTS/poem/unet2 RESULTS/poem/deepmed2 RESULTS/poem/pnet2
TRN3 = RESULTS/poem/unet_dts3d RESULTS/poem/deepmed_dts3d RESULTS/poem/pnet_dts3d
TRN4 = RESULTS/poem/unet_dts1 RESULTS/poem/deepmed_dts1 RESULTS/poem/pnet_dts1 RESULTS/poem/unet_dts2 RESULTS/poem/deepmed_dts2 RESULTS/poem/pnet_dts2

RESULTS/poem/unet1: SV = unet
RESULTS/poem/unet1: OPT = --losses="[('WeightedGeneralizedDice', {'idc': [0.5, 1, 1, 1, 1, 1, 1]}, 1), \
					('WeightedCrossEntropy', {'idc': [0.1, 1, 1, 1, 1, 1, 1]}, 1)]" \
					"--save_as=$(SV)" '--network=UNet' '--schedule' $(DATA1)
RESULTS/poem/unet2: SV = unet
RESULTS/poem/unet2: OPT = --losses="[('WeightedGeneralizedDice', {'idc': [0.5, 1, 1, 1, 1, 1, 1]}, 1), \
					('WeightedCrossEntropy', {'idc': [0.1, 1, 1, 1, 1, 1, 1]}, 1)]" \
					"--save_as=$(SV)" '--network=UNet' '--schedule' $(DATA2)


RESULTS/poem/deepmed1: SV = deepmed
RESULTS/poem/deepmed1: OPT = --losses="[('WeightedGeneralizedDice', {'idc': [0.5, 1, 1, 1, 1, 1, 1]}, 1), \
					('WeightedCrossEntropy', {'idc': [0.1, 1, 1, 1, 1, 1, 1]}, 1)]" \
					"--save_as=$(SV)" '--network=DeepMedic' '--schedule' $(DATA1)
RESULTS/poem/deepmed2: SV = deepmed
RESULTS/poem/deepmed2: OPT = --losses="[('WeightedGeneralizedDice', {'idc': [0.5, 1, 1, 1, 1, 1, 1]}, 1), \
					('WeightedCrossEntropy', {'idc': [0.1, 1, 1, 1, 1, 1, 1]}, 1)]" \
					"--save_as=$(SV)" '--network=DeepMedic' '--schedule' $(DATA2)


RESULTS/poem/pnet1: SV = pnet
RESULTS/poem/pnet1: OPT = --losses="[('WeightedGeneralizedDice', {'idc': [0.5, 1, 1, 1, 1, 1, 1]}, 1), \
					('WeightedCrossEntropy', {'idc': [0.1, 1, 1, 1, 1, 1, 1]}, 1)]" \
					"--save_as=$(SV)" '--network=PSPNet' '--schedule' $(DATA1)
RESULTS/poem/pnet2: SV = pnet
RESULTS/poem/pnet2: OPT = --losses="[('WeightedGeneralizedDice', {'idc': [0.5, 1, 1, 1, 1, 1, 1]}, 1), \
					('WeightedCrossEntropy', {'idc': [0.1, 1, 1, 1, 1, 1, 1]}, 1)]" \
					"--save_as=$(SV)" '--network=PSPNet' '--schedule' $(DATA2)


RESULTS/poem/unet3d: SV = unet
RESULTS/poem/unet3d: OPT = --losses="[('WeightedGeneralizedDice', {'idc': [0.5, 1, 1, 1, 1, 1, 1]}, 1), \
					('WeightedCrossEntropy', {'idc': [0.1, 1, 1, 1, 1, 1, 1]}, 1)]" \
					"--save_as=$(SV)" '--network=UNet' '--schedule' '--in3D' $(DATA)


RESULTS/poem/deepmed3d: SV = deepmed
RESULTS/poem/deepmed3d: OPT = --losses="[('WeightedGeneralizedDice', {'idc': [0.5, 1, 1, 1, 1, 1, 1]}, 1), \
					('WeightedCrossEntropy', {'idc': [0.1, 1, 1, 1, 1, 1, 1]}, 1)]" \
					"--save_as=$(SV)" '--network=DeepMedic' '--schedule' '--in3D' $(DATA)


RESULTS/poem/pnet3d: SV = pnet
RESULTS/poem/pnet3d: OPT = --losses="[('WeightedGeneralizedDice', {'idc': [0.5, 1, 1, 1, 1, 1, 1]}, 1), \
					('WeightedCrossEntropy', {'idc': [0.1, 1, 1, 1, 1, 1, 1]}, 1)]" \
					"--save_as=$(SV)" '--network=PSPNet' '--schedule' '--in3D' $(DATA)



RESULTS/poem/unet_dts1: SV = unet
RESULTS/poem/unet_dts1: OPT = --losses="[('WeightedGeneralizedDice', {'idc': [0.5, 1, 1, 1, 1, 1, 1]}, 1), \
					('WeightedCrossEntropy', {'idc': [0.1, 1, 1, 1, 1, 1, 1]}, 1)]" \
					"--save_as=$(SV)" '--network=UNet' '--schedule' $(DATA1)
RESULTS/poem/unet_dts2: SV = unet
RESULTS/poem/unet_dts2: OPT = --losses="[('WeightedGeneralizedDice', {'idc': [0.5, 1, 1, 1, 1, 1, 1]}, 1), \
					('WeightedCrossEntropy', {'idc': [0.1, 1, 1, 1, 1, 1, 1]}, 1)]" \
					"--save_as=$(SV)" '--network=UNet' '--schedule' $(DATA2)

RESULTS/poem/pnet_dts1: SV = pnet
RESULTS/poem/pnet_dts1: OPT = --losses="[('WeightedGeneralizedDice', {'idc': [0.5, 1, 1, 1, 1, 1, 1]}, 1), \
					('WeightedCrossEntropy', {'idc': [0.1, 1, 1, 1, 1, 1, 1]}, 1)]" \
					"--save_as=$(SV)" '--network=PSPNet' '--schedule' $(DATA1)
RESULTS/poem/pnet_dts2: SV = pnet
RESULTS/poem/pnet_dts2: OPT = --losses="[('WeightedGeneralizedDice', {'idc': [0.5, 1, 1, 1, 1, 1, 1]}, 1), \
					('WeightedCrossEntropy', {'idc': [0.1, 1, 1, 1, 1, 1, 1]}, 1)]" \
					"--save_as=$(SV)" '--network=PSPNet' '--schedule' $(DATA2)

RESULTS/poem/deepmed_dts1: SV = deepmed
RESULTS/poem/deepmed_dts1: OPT = --losses="[('WeightedGeneralizedDice', {'idc': [0.5, 1, 1, 1, 1, 1, 1]}, 1), \
					('WeightedCrossEntropy', {'idc': [0.1, 1, 1, 1, 1, 1, 1]}, 1)]" \
					"--save_as=$(SV)" '--network=DeepMedic' '--schedule' $(DATA1)
RESULTS/poem/deepmed_dts2: SV = deepmed
RESULTS/poem/deepmed_dts2: OPT = --losses="[('WeightedGeneralizedDice', {'idc': [0.5, 1, 1, 1, 1, 1, 1]}, 1), \
					('WeightedCrossEntropy', {'idc': [0.1, 1, 1, 1, 1, 1, 1]}, 1)]" \
					"--save_as=$(SV)" '--network=DeepMedic' '--schedule' $(DATA2)



RESULTS/poem/unet_dts3d: SV = unet
RESULTS/poem/unet_dts3d: OPT = --losses="[('WeightedGeneralizedDice', {'idc': [0.5, 1, 1, 1, 1, 1, 1]}, 1), \
					('WeightedCrossEntropy', {'idc': [0.1, 1, 1, 1, 1, 1, 1]}, 1)]" \
					"--save_as=$(SV)" '--network=UNet' '--schedule' '--in3D' $(DATA)

RESULTS/poem/pnet_dts3d: SV = pnet
RESULTS/poem/pnet_dts3d: OPT = --losses="[('WeightedGeneralizedDice', {'idc': [0.5, 1, 1, 1, 1, 1, 1]}, 1), \
					('WeightedCrossEntropy', {'idc': [0.1, 1, 1, 1, 1, 1, 1]}, 1)]" \
					"--save_as=$(SV)" '--network=PSPNet' '--schedule' '--in3D' $(DATA)

RESULTS/poem/deepmed_dts3d: SV = deepmed
RESULTS/poem/deepmed_dts3d: OPT = --losses="[('WeightedGeneralizedDice', {'idc': [0.5, 1, 1, 1, 1, 1, 1]}, 1), \
					('WeightedCrossEntropy', {'idc': [0.1, 1, 1, 1, 1, 1, 1]}, 1)]" \
					"--save_as=$(SV)" '--network=DeepMedic' '--schedule' '--in3D' $(DATA)


all: $(TRN1) $(TRN2) $(TRN3) $(TRN4)
$(TRN1) $(TRN2):
	mkdir -p $@_tmp
	$(CC) $(CFLAGS) Training.py --batch_size=32 --l_rate=1e-3 \
		--n_epoch=$(EPC) --in_channels 0 1 --lower_in_channels 0 1 $(OPT) 
	mv $@_tmp $@ 
	mv RESULTS/$(SV)* $@/.
	

$(TRN3) $(TRN4):
	mkdir -p $@_tmp
	$(CC) $(CFLAGS) Training.py --batch_size=32 --l_rate=1e-3 \
		--n_epoch=$(EPC) --in_channels 0 1 2 3 4 --lower_in_channels 0 1 5 $(OPT)
	mv $@_tmp $@ 
	mv RESULTS/$(SV)* $@/.


