#!/bin/sh
python sreening.py \
--protein_ligand_csv data/kor_Ki_prot_lig.csv \
--model_dir workdir/big_score_model_sanyueqi_with_time \
--savings_per_complex 1 \
--inference_steps 20 \

