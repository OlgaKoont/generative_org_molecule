import os
from multiprocessing import Pool
from subprocess import run
import glob
import numpy as np
import pandas as pd
from tempfile import NamedTemporaryFile


class DBAffinity:
    def __init__(self, config):
        self.config = config

    def __call__(self, smile_list):
        os.environ['PATH'] = os.path.dirname(self.config['relax_python']) + ":" + os.environ['PATH']
        file_path = os.path.realpath(__file__)
        script_folder = os.path.dirname(file_path)
        os.makedirs("data", exist_ok=True)

        if {self.config['paper']}:
            model_workdir = f"{script_folder}/workdir/big_score_model_sanyueqi_with_time"
            ckpt = "ema_inference_epoch314_model.pt"
        else:
            if model == 1:
                model_workdir = f"{script_folder}/workdir/big_score_model_sanyueqi_with_time"
                ckpt = "pro_ema_inference_epoch138_model.pt"

        if not {self.config['rigid_protein']}:
            protein_dynamic = "--protein_dynamic"
        else:
            protein_dynamic = ""

        with NamedTemporaryFile(mode='r+t', suffix='.csv', delete=False) as f1, NamedTemporaryFile(mode='r+t', suffix='.fasta', delete=False) as f2:            
            pd.DataFrame({'ligand': smile_list, 'protein_path': [self.config['protein_pdb']] * len(smile_list)}).to_csv(f1.name)

            cmd1 = f"{self.config['python']} {script_folder}/datasets/esm_embedding_preparation.py"
            cmd1 += f" --protein_ligand_csv {f1.name} --out_file {f2.name}"
            result1 = run(cmd1.split(), capture_output=True, text=True, timeout=90)
            if result1.stdout:
                print("\nstdout1:")
                print(result1.stdout)
            if result1.stderr:
                print("\nstErrrs1:")
                print(result1.stderr)

            cmd2 = f"env MKL_SERVICE_FORCE_INTEL=1 MKL_THREADING_LAYER=GNU CUDA_VISIBLE_DEVICES={self.config['device']}"
            cmd2 += f" {self.config['python']} {script_folder}/esm/scripts/extract.py esm2_t33_650M_UR50D" 
            cmd2 += f" {f2.name} data/esm2_output --repr_layers 33" 
            cmd2 += f" --include per_tok --truncation_seq_length 10000 --model_dir {script_folder}/esm_models"
            result2 = run(cmd2.split(), capture_output=True, text=True)
            if result2.stdout:
                print("\nstdout2:")
                print(result2.stdout)
            if result2.stderr:
                print("\nstErrrs2:")
                print(result2.stderr)

            cmd3 = f"env MKL_SERVICE_FORCE_INTEL=1 MKL_THREADING_LAYER=GNU CUDA_VISIBLE_DEVICES={self.config['device']}" 
            cmd3 += f" {self.config['python']} {script_folder}/screening_airi.py --seed {self.config['seed']} --ckpt {ckpt} {protein_dynamic}"
           #cmd3 += f" --relax_python {self.config['relax_python']}"
            cmd3 += f" --save_visualisation --model_dir {model_workdir}"
            cmd3 += f" --protein_ligand_csv {f1.name}" #
            cmd3 += " --esm_embeddings_path data/esm2_output" 
            cmd3 += f" --inference_steps {self.config['inference_steps']}"
            cmd3 += f" --out_dir {self.config['results']}/{self.config['header']}"
            cmd3 += f" --inference_steps {self.config['inference_steps']}"
            cmd3 += " --savings_per_complex 0" 
            cmd3 += " --samples_per_complex 1"
            cmd3 += f" --batch_size_db {self.config['batch_size']}"
            cmd3 += f" --actual_steps {self.config['inference_steps']} "
            cmd3 += " --no_final_step_noise"
            result3 = run(cmd3.split(), capture_output=True, text=True)
            if result3.stdout:
                print("\nstdout3:")
                print(result3.stdout)
            if result3.stderr:
                print("\nstErrrs3:")
                print(result3.stderr)
        path = os.path.join(self.config['results'], self.config['header'], 'affinity_prediction.csv')
        if os.path.exists(path):
            affinities = pd.read_csv(path)['affinity'].tolist()
        else:
            affinities = [0] * len(smile_list)
        try:
            os.remove(f1.name)
            os.remove(f2.name)
        except:
            pass
        return affinities
