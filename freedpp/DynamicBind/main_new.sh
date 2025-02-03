#!/bin/sh
python run_single_protein_inference.py \
--proteinFile data/4djh_cleaned.pdb \
--ligandFile ligand.csv \
--savings_per_complex 1 \
--inference_steps 20 \
--header main_new \
--results result_folder_print \
--python /nfs/home/okonovalova/anaconda3/envs/dynamicbind/bin/python \
--relax_python /nfs/home/okonovalova/anaconda3/envs/relax/bin/python