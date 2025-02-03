#!/bin/sh
python run_single_protein_inference.py \
--proteinFile data/4djh_cleaned.pdb \
--ligandFile ligand.csv \
--savings_per_complex 1 \
--inference_steps 20 \
--header test_0102 \
--results result_folder \
--python /nfs/home/okonovalova/anaconda3/envs/dynamicbind/bin/python \
--relax_python /nfs/home/okonovalova/anaconda3/envs/relax/bin/python