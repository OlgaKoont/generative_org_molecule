#!/bin/sh
python ../main_select_two.py \
	    --exp_root experiments \
	        --alert_collections ../alert_collections.csv \
		    --fragments ../zinc_crem.json \
                        --receptor_1 ../protein_4djh_chain_A.pdbqt \
		        --receptor_2 ../protein_8j2o.pdbqt \
                           --vina_program ../env/qvina02 \
			   --starting_smile "c1([*:1])c([*:2])ccc([*:3])c1" \
			   --fragmentation crem \
			   --num_sub_proc 12 \
			   --n_conf 1 \
                           --num_mols 1000 \
			   --exhaustiveness 1 \
			   --save_freq 50 \
			   --epochs 200 \
			   --commands "train,sample" \
			   --reward_version soft \
                           --box_center_2 "16.357,4.471,13.989" \
                           --box_center_1 "4.139,-22.228,60.007" \
    		           --box_size_2 "10.366,14.254,17.264" \
                           --box_size_1 "17.206,25.0,20.574" \
			   --seed 42 \
                           --weights "1.0,1.0" \
                           --objectives "DockingScore_1,DockingScore_2" \
                           --name select_for_4djh_8j2o_42 \
                           #--checkpoint experiments/select_for_4djh_8j2o_42/ckpt/model_200.pth
