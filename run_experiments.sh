#!/bin/bash
n=100
ncoeff=0.2
for mech in "nn" "gp_add" "gp_mix" 
do
for s in 10
do 
mkdir -p experiments/${mech}_n${n}_s${s}_ncoeff${ncoeff} 
for rep in {1..5}
do
    tmux new-session -d "python3 train_test_gen.py --ntrain ${n} --ntest ${n} --mech ${mech} -s ${s} --noise-coeff ${ncoeff} --rescale -o experiments/${mech}_n${n}_s${s}_ncoeff${ncoeff}/run${rep}.csv"
done
done
done
