#!/bin/bash

for net in VanillaCNN
do
	if [ "$net" = "UNet" ]; then
		out=unet 
	else 
		out=van
	fi
	echo "Evaluating network $net..."
	for rep in 1 2 3 4 5
	do
		echo " * Repetition nr.$rep"
		python3 FoldEval.py --path="$net/rep$rep/${out}_1f/$out"
	done
done
