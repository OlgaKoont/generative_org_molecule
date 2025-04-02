#!/bin/bash
python ffreed/main_airi.py --input_file ../AIRI_project/datasets/dynamic_cyp2d6_ki.csv \
            --output_file ../AIRI_project/results/4wnv/dynamic_cyp2d6_ki_out.csv \
            --savings_per_complex 1 \
            --db_inference_steps 20 \
            --db_batch_size 16 \
            --db_device 1 \
            --seed 42 \
            --db_protein_pdb ../AIRI_project/proteins/4wnv_meeko.pdb \
            --db_header cyp_2d6_ki \
            --db_result AIRI_project/results/4wnv/ \
            --esm2_output esm2_output \
            --db_env /nfs/home/okonovalova/miniconda3/envs/dynamicbind/bin/python \
            --db_relax_env /nfs/home/okonovalova/miniconda3/envs/relax/bin/python \


python ffreed/main_airi.py --input_file ../AIRI_project/datasets/dynamic_cyp3a4_ki.csv \
            --output_file ../AIRI_project/results/1dpf/dynamic_cyp3a4_ki_out.csv \
            --savings_per_complex 1 \
            --db_inference_steps 20 \
            --db_batch_size 16 \
            --db_device 1 \
            --seed 42 \
            --db_protein_pdb ../AIRI_project/proteins/1dpf_meeko.pdb \
            --db_header cyp_cyp3a4_ki \
            --db_result AIRI_project/results/1dpf/ \
            --esm2_output esm2_output_1dpf \
            --db_env /nfs/home/okonovalova/miniconda3/envs/dynamicbind/bin/python \
            --db_relax_env /nfs/home/okonovalova/miniconda3/envs/relax/bin/python \
