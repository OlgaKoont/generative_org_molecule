#!/nfs/home/okonovalova/anaconda3/envs/dynamicbind/bin/python 
import numpy as np
import pandas as pd

import time
import os
import sys
import subprocess
from datetime import datetime
import logging
import rdkit.Chem as Chem
import torch
torch.cuda.empty_cache()


import os
from multiprocessing import Pool
from subprocess import run
import glob
import numpy as np
from tempfile import NamedTemporaryFile

timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")

class DB:
    def __init__(self, config):
        self.config = config

    def __call__(self, smile):
        os.environ['OB_RANDOM_SEED'] = str(self.config['seed'])
        print(DB.run_inference(smile, **self.config)) 
        return DB.run_inference(smile, **self.config)  
    
    def do(self, cmd, get=False, show=True):
        if get:
            out = subprocess.Popen(cmd,stdout=subprocess.PIPE,shell=True).communicate()[0].decode()
            if show:
                print(out, end="")
            return out
        else:
            return subprocess.Popen(cmd, shell=True).wait()


    logging.basicConfig(level=logging.INFO)
    handler = logging.FileHandler(f'run.log')
    logger = logging.getLogger("")
    logger.addHandler(handler)

    logging.info(f'''\
    {' '.join(sys.argv)}
    {timestamp}
    --------------------------------
    ''')

    # python='/mnt/nas/glx-share-cache/InfraDev/glx-schrodinger/envs/dynamicbind_rdkit2022/bin/python'

    @staticmethod
    def run_inference(smile, *, python, relax_python, results, proteinFile, num_workers,
                      header, paper, model, rigid_protein, hts, device, seed, n_conf,
                      no_inference, no_relax, samples_per_complex, batch_size, 
                      savings_per_complex, inference_steps, no_clean, **kwargs):

        os.environ['PATH'] = os.path.dirname(relax_python) + ":" + os.environ['PATH']
        file_path = os.path.realpath(__file__)
        script_folder = os.path.dirname(file_path)
        print(file_path, script_folder)
        os.makedirs("data", exist_ok=True)

    
#       cleaned_proteinFile = "./data/cleaned_input_proteinFile.pdb"
        ligandFile_with_protein_path = f"./data/ligandFile_with_protein_path_{timestamp}.csv"
#       cmd = f"{relax_python} {script_folder}/clean_pdb.py {proteinFile} {cleaned_proteinFile}"
#       self.do(cmd)
        print(smile)
        print(proteinFile)
        ligands = pd.DataFrame({'ligand': [smile], 'protein_path': [proteinFile]})
        assert 'ligand' in ligands.columns
        ligands.to_csv(ligandFile_with_protein_path, index=0)

    
        if paper:
            model_workdir = f"{script_folder}/workdir/big_score_model_sanyueqi_with_time"
            ckpt = "ema_inference_epoch314_model.pt"
        else:
            if model == 1:
                model_workdir = f"{script_folder}/workdir/big_score_model_sanyueqi_with_time"
                ckpt = "pro_ema_inference_epoch138_model.pt"

        if not rigid_protein:
            protein_dynamic = "--protein_dynamic"
        else:
            protein_dynamic = ""
    
        t0 = time.time()
        if hts:
            os.system("mkdir -p data")
            run_line = f"{python} {script_folder}/datasets/esm_embedding_preparation.py --protein_ligand_csv {ligandFile_with_protein_path} --out_file data/prepared_for_esm_{header}.fasta"
            run(run_line.split(), shell=True, capture_output=True, text=True)            
            run_line = f"MKL_SERVICE_FORCE_INTEL=1 MKL_THREADING_LAYER=GNU CUDA_VISIBLE_DEVICES={device} {python} {script_folder}/esm/scripts/extract.py esm2_t33_650M_UR50D data/prepared_for_esm_{header}.fasta data/esm2_output --repr_layers 33 --include per_tok --truncation_seq_length 10000 --model_dir {script_folder}/esm_models"
            run(run_line.split(), shell=True, capture_output=True, text=True)
            run_line = f"MKL_SERVICE_FORCE_INTEL=1 MKL_THREADING_LAYER=GNU CUDA_VISIBLE_DEVICES={device} {python} {script_folder}/screening_old.py --seed {seed} --ckpt {ckpt} {protein_dynamic}"
            run_line += f" --save_visualisation --model_dir {model_workdir}  --protein_ligand_csv {ligandFile_with_protein_path} "
            run_line += f" --esm_embeddings_path data/esm2_output --out_dir {results}/{header} --inference_steps {inference_steps} --samples_per_complex {samples_per_complex} --savings_per_complex {savings_per_complex} --batch_size {batch_size} --actual_steps {inference_steps} --no_final_step_noise"
            run(run_line.split(), shell=True, capture_output=True, text=True)
        else:
            if not no_inference:
                os.system("mkdir -p data")
                run_line = f"{python} {script_folder}/datasets/esm_embedding_preparation.py --protein_ligand_csv {ligandFile_with_protein_path} --out_file data/prepared_for_esm_{header}.fasta"
                run(run_line.split(), shell=True, capture_output=True, text=True)
                run_line = f"CUDA_VISIBLE_DEVICES={device} {python} {script_folder}/esm/scripts/extract.py esm2_t33_650M_UR50D data/prepared_for_esm_{header}.fasta data/esm2_output --repr_layers 33 --include per_tok --truncation_seq_length 10000 --model_dir {script_folder}/esm_models"
                run(run_line.split(), shell=True, capture_output=True, text=True)
                run_line = f"CUDA_VISIBLE_DEVICES={device} {python} {script_folder}/inference.py --seed {seed} --ckpt {ckpt} {protein_dynamic}"
                run_line += f" --save_visualisation --model_dir {model_workdir}  --protein_ligand_csv {ligandFile_with_protein_path} "
                run_line += f" --esm_embeddings_path data/esm2_output --out_dir {results}/{header} --inference_steps {inference_steps} --samples_per_complex {samples_per_complex} --savings_per_complex {savings_per_complex} --batch_size {batch_size} --actual_steps {inference_steps} --no_final_step_noise"
                run(run_line.split(), shell=True, capture_output=True, text=True)
                print("inference complete.")

            if not no_relax:
                run_line = f"CUDA_VISIBLE_DEVICES={device} {relax_python} {script_folder}/relax_final.py --results_path {results}/{header} --samples_per_complex {samples_per_complex} --num_workers {num_workers}"
                # print("relax final step structure.")
                # exit()
                run(run_line.split(), shell=True, capture_output=True, text=True)
                print("final step structure relax complete.")
        
        t1 = time.time()
        print('time', t1 - t0)
        file_path = f'/mnt/tank/scratch/okonovalova/freedpp/freedpp/dynamicbind/{results}/{header}/affinity_prediction.csv'  
        df = pd.read_csv(file_path)
        result = float(df['affinity'][0])
        return result





  #      if movie:
  #          for i in range(len(ligands)):
   #             cmd = f"CUDA_VISIBLE_DEVICES={device} {relax_python} {script_folder}/movie_generation.py {results}/{header}/index{i}_idx_{i} 1 --python {python} --relax_python {relax_python} --inference_steps {inference_steps}"
     #           self.do(cmd)
    #            print(cmd)