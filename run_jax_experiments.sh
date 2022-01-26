#!/bin/bash

ncoeff=0.4
for mech in "nn" "sigmoid_add" "sigmoid_mix" do
for n in 100 200 300 400 500 do  
for s in 100 do 
mkdir -p jax_experiments/${mech}_n${n}_s${s}_ncoeff${ncoeff} 
for rep in {1..5} do
    python jax_train_test_gen.py --ntrain ${n} --ntest ${n} --mech ${mech} -s ${s} --noise-coeff ${ncoeff} --rescale -o jax_experiments/${mech}_n${n}_s${s}_ncoeff${ncoeff}/run${rep}.csv
done
done
done
done 
