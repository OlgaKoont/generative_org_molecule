#!/bin/bash
python ffreed/main_airi_02.py --input_file ../datasets/db_2d6_ki_52019_8.csv \
            --output_file ../results/4wnv/db_2d6_ki_52019_8_out.csv \
            --savings_per_complex 1 \
            --db_inference_steps 20 \
            --db_batch_size 5 \
            --db_device 1 \
            --seed 42 \
            --db_protein_pdb ../proteins/4wnv_meeko.pdb \
            --db_header db_2d6_ki_52019_8 \
            --db_result results/4wnv/ \
            --esm2_output esm2_output \
            --db_env /nfs/home/okonovalova/miniconda3/envs/dynamicbind/bin/python \
            --db_relax_env /nfs/home/okonovalova/miniconda3/envs/relax/bin/python \



