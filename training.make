CC = python3.6
SHELL = /usr/bin/zsh

EPC = 120


DATA3D = --dataset='POEM25_1'
DATA3D_2 = --dataset='POEM25_2'
DATA3D_3 = --dataset='POEM25_3'

DATA3D_35 = --dataset='POEM35_3'
#DATA = --dataset='POEM25'

VERS =_4

TRN3D = RESULTS/un25 RESULTS/un35 RESULTS/un25s RESULTS/un35s # RESULTS/un_6 RESULTS/un_8 #RESULTS/un3d$(VERS) #RESULTS/dm3d$(VERS) RESULTS/pn3d$(VERS) 
#TRN = RESULTS/un$(VERS) RESULTS/dm$(VERS) RESULTS/pn$(VERS)
TRN3D_DT = RESULTS/un25_dt RESULTS/un35_dt RESULTS/un25s_dt RESULTS/un35s_dt #RESULTS/un3d_dt$(VERS) #RESULTS/dm3d_dt$(VERS) RESULTS/pn3d_dt$(VERS)
#TRN_DT = RESULTS/un_dt$(VERS) RESULTS/dm_dt$(VERS) RESULTS/pn_dt$(VERS)



all: $(TRN3D) $(TRN3D_DT) #$(TRN) $(TRN_DT) 

RESULTS/un25% RESULTS/un35%: CE=[0.5, 1,1,1,1,1,1]
RESULTS/un25%: DATA=--dataset='POEM25_3'
RESULTS/un35%: DATA=--dataset='POEM35_3'
RESULTS/un25s% RESULTS/un35s%: NET='--network=UNetShallow'
RESULTS/un25 RESULTS/un25_dt RESULTS/un35_dt RESULTS/un35: NET='--network=UNet'

RESULTS/un%: SV = unet
RESULTS/un%: OPT = --losses="[('WeightedGeneralizedDice', {'idc': [0.01, 0.25, 0.16, 0.05, 0.34, 0.11, 0.16]}, 1), \
					('WeightedCrossEntropy', {'idc': $(CE)}, 1)]" \
					"--save_as=$(SV)" '--schedule' $(NET)
#'--network=UNet'


#CE: [0.01, 0.3, 0.2, 0.01, 0.4, 0.15, 0.2]
RESULTS/dm%: SV = deepmed
RESULTS/dm%: OPT = --losses="[('WeightedGeneralizedDice', {'idc': [0.01, 0.25, 0.16, 0.05, 0.34, 0.11, 0.16]}, 1), \
					('WeightedCrossEntropy', {'idc': [0.01, 0.25, 0.16, 0.05, 0.34, 0.11, 0.16]}, 1)]" \
					"--save_as=$(SV)" '--network=DeepMedic' '--schedule'


RESULTS/pn%: SV = pnet
RESULTS/pn%: OPT = --losses="[('WeightedGeneralizedDice', {'idc': [0.01, 0.25, 0.16, 0.05, 0.34, 0.11, 0.16]}, 1), \
					('WeightedCrossEntropy', {'idc': [0.01, 0.25, 0.16, 0.05, 0.34, 0.11, 0.16]}, 1)]" \
					"--save_as=$(SV)" '--network=PSPNet' '--schedule'

#('WeightedCrossEntropy', {'idc': [0.5, 1, 1, 1, 1, 1, 1]}, 1) pri 3D, razen pri dm

#RESULTS/%_dt$(VERS): IN1 = 2 3 5
#RESULTS/%_dt$(VERS): IN2 = 5
RESULTS/%_dt: IN1 = 2 3 5
RESULTS/%_dt: IN2 = 5


#in2D:
$(TRN) $(TRN_DT):
	mkdir -p $@_tmp
	$(CC) $(CFLAGS) Training.py --batch_size=32 --l_rate=1e-3 \
		--n_epoch=$(EPC) --in_channels 0 1 $(IN1) --lower_in_channels 0 1 $(IN2) $(OPT) $(DATA)
	mv $@_tmp $@ 
	mv RESULTS/$(SV)* $@/.
	
#in 3D:
$(TRN3D) $(TRN3D_DT):
	mkdir -p $@_tmp
	$(CC) $(CFLAGS) Training.py --batch_size=32 --l_rate=1e-3 \
		--n_epoch=$(EPC) --in_channels 0 1 $(IN1) --lower_in_channels 0 1 $(IN2) $(OPT) \
		--in3D $(DATA)
	mv $@_tmp $@ 
	mv RESULTS/$(SV)* $@/.
# --restore_from="$@_old/$(SV)"