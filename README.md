`smm.py` is the basic Support measure machine classifier it works with any SVM method from sklearn and any kernel passed as callable (check try_smm.py for examples)

`meta_causal_smm.py` is the meta method which builds one smm classifier per base algorithms (given as dictionary) and then average base algorithms decisions with the smm decision functions (depends on smm.py


`try_meta_learner.py` is an example script for testing the meta method plus the base alg and pure smm, it accepts command line arguments to specify mechanism ntrain ntest and sample size

`train_test_gen.py` is similar to `try_meta_learner.py` but it saves the results to a csv files (name can be passed via -o outfile.csv ) and computes execution times

`run_experiment.sh` is a bash script that runs a lot of experiments using tmux, to be used mainly in the server but also locally but carefully if you do not want to use all your cpus

`train_gen_test_tub.py` is to train on generated data and test on tubingen, but ANM was not working on tuebingen

only requirements are `sklearn , numpy, pandas, cdt`
