#!/bin/sh
python ../main_qed.py \
	    --exp_root experiments \
	        --alert_collections ../alert_collections.csv \
		    --fragments ../zinc_crem.json \
		        --receptor ../protein_A_6vn3.pdbqt \
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
                           --box_center "28.111,0.895,29.799" \
    		           --box_size "16.442,17.990,19.716" \
			   --seed 150 \
                           --weights "1.0,1.0" \
                           --objectives "DockingScore,QED" \
                           --name freedpp_qed_s_150 \
                           #--checkpoint experiments/freedpp_qed_s_150/ckpt/model_200.pth

                           

