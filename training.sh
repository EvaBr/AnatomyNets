#!/bin/bash



epc=120
opt="[('WeightedGeneralizedDice', {'idc': [0.01, 0.25, 0.16, 0.05, 0.34, 0.12, 0.16]}, 1), \
('WeightedCrossEntropy', {'idc': [0.5, 1,1,1.1,1,1,1]}, 1)]"
		
for net in UNet VanillaCNN
do
	if [ "$net" = "UNet" ]; then
		out=unet 
	else 
		out=van
	fi
	echo "Doing network $net..."
	for rep in 1 2 3 4 5
	do
		echo "    Repetition nr.$rep"
		for fold in 1 2 3 4 5
		do
			echo "        Fold $fold"
			#ORIGINAL
			echo "        * original"
			path="RESULTS/$net/rep$rep/${out}_${fold}f"
			mkdir -p "${path}_tmp"
			python3 Training.py --l_rate=1e-3 --n_epoch=$epc \
					--in_channels 0 1 --lower_in_channels 0 1 --losses="$opt" --schedule \
					--batch_size=32 --save_as=$out --network=$net \
					--in3D --dataset="POEM_WB$fold"
			mv "${path}_tmp" "$path" 
			mv "RESULTS/$out"* "$path/."

			#WITH DTs
			echo "        * with DTs"
			path="RESULTS/$net/rep$rep/${out}_${fold}f_dt"
			mkdir -p "${path}_tmp"
			python3 Training.py --l_rate=1e-3 --n_epoch=$epc \
					--in_channels 0 1 2 3 5 --lower_in_channels 0 1 --losses="$opt" --schedule \
					--batch_size=32 --save_as=$out --network=$net \
					--in3D --dataset="POEM_WB$fold"
			mv "${path}_tmp" "$path" 
			mv "RESULTS/$out"* "$path/."
		done
	done
done


			
					

#CE: [0.01, 0.3, 0.2, 0.01, 0.4, 0.15, 0.2]
#RESULTS/dm%: SV = deepmed
#RESULTS/dm%: OPT = --losses="[('WeightedGeneralizedDice', {'idc': [0.01, 0.25, 0.16, 0.05, 0.34, 0.11, 0.16]}, 1), \
#					('WeightedCrossEntropy', {'idc': [0.01, 0.25, 0.16, 0.05, 0.34, 0.11, 0.16]}, 1)]" \
#					"--save_as=$(SV)" '--network=DeepMedic' '--schedule'



