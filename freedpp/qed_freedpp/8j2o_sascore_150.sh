#!/bin/sh
python ../main_qed_sascore.py \
	    --exp_root experiments \
	        --alert_collections ../alert_collections.csv \
		    --fragments ../zinc_crem.json \
		        --receptor ../protein_8j2o.pdbqt \
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
                            --box_center "16.357,4.471,13.989" \
    		           --box_size "10.366,14.254,17.264" \
			   --seed 150 \
                           --weights "1.0,1.0,1.0" \
                           --objectives "DockingScore,SA_Score,QED" \
                           --name 8j2o_sascore_qed_150 \
                           #--checkpoint experiments/8j2o_sascore_qed_150/ckpt/model_200.pth


                           

