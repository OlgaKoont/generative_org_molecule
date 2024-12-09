#!/bin/sh
python ../main_select_two.py \
	    --exp_root experiments \
	        --alert_collections ../alert_collections.csv \
		    --fragments ../zinc_crem.json \
                        --receptor_1 ../protein_4djh_chain_A.pdbqt \
		        --receptor_2 ../protein_8efo_chain_R.pdbqt \
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
                           --box_center_2 "185.111,172.052,173.091" \
                           --box_center_1 "4.139,-22.228,60.007" \
    		           --box_size_2 "15.202,21.452,14.052" \
                           --box_size_1 "17.206,25.0,20.574" \
			   --seed 42 \
                           --weights "1.0,1.0" \
                           --objectives "DockingScore_1,DockingScore_2" \
                           --name select_kor_mor_42 \
                           #--checkpoint experiments/select_kor_mor_42/ckpt/model_200.pth

