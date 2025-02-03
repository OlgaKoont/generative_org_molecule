#!/nfs/home/okonovalova/anaconda3/envs/dynamicbind/bin/python \
import numpy as np
import pandas as pd

import time
import os
import sys
import subprocess
from datetime import datetime
import logging
import rdkit.Chem as Chem

def do(cmd, get=False, show=True):
    if get:
        out = subprocess.Popen(cmd,stdout=subprocess.PIPE,shell=True).communicate()[0].decode()
        if show:
            print(out, end="")
        return out
    else:
        return subprocess.Popen(cmd, shell=True).wait()


import argparse

timestamp = datetime.now().strftime("%Y_%m_%d_%H_%M")

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



def run_inference(args):
    print(args)
    python = args['python']
    relax_python = args['relax_python']

    os.environ['PATH'] = os.path.dirname(relax_python) + ":" + os.environ['PATH']
    file_path = os.path.realpath(__file__)
    script_folder = os.path.dirname(file_path)
    print(file_path, script_folder)
    os.makedirs("data", exist_ok=True)

    
    if args['protein_path_in_ligandFile']:
        if args['no_clean']:
            ligandFile_with_protein_path = args['ligandFile']
        else:
            ligandFile_with_protein_path = f"./data/ligandFile_with_protein_path_{datetime.now().strftime('%Y_%m_%d_%H_%M')}.csv"
            cmd = f"{relax_python} {script_folder}/clean_pdb.py {args['ligandFile']} {ligandFile_with_protein_path}"
            do(cmd)

        ligands = pd.read_csv(ligandFile_with_protein_path)
        assert 'ligand' in ligands.columns
        assert 'protein_path' in ligands.columns

    elif args['ligand_is_sdf']:
        cleaned_proteinFile = "./data/cleaned_input_proteinFile.pdb"
        ligandFile_with_protein_path = f"./data/ligandFile_with_protein_path_{datetime.now().strftime('%Y_%m_%d_%H_%M')}.csv"
        cmd = f"{relax_python} {script_folder}/clean_pdb.py {args['proteinFile']} {cleaned_proteinFile}"
        do(cmd)

        ligandFile = "./data/" + os.path.basename(args['ligandFile'])
        mol = Chem.MolFromMolFile(args['ligandFile'])
        _ = Chem.MolToSmiles(mol)
        m_order = list(mol.GetPropsAsDict(includePrivate=True, includeComputed=True)["_smilesAtomOutputOrder"])
        mol = Chem.RenumberAtoms(mol, m_order)
        
        w = Chem.SDWriter(ligandFile)
        w.write(mol)
        w.close()
        
        ligands = pd.DataFrame({"ligand": [ligandFile], "protein_path": [cleaned_proteinFile]})
        ligands.to_csv(ligandFile_with_protein_path, index=0)

    else:
        cleaned_proteinFile = "./data/cleaned_input_proteinFile.pdb"
        ligandFile_with_protein_path = f"./data/ligandFile_with_protein_path_{datetime.now().strftime('%Y_%m_%d_%H_%M')}.csv"
        cmd = f"{relax_python} {script_folder}/clean_pdb.py {args['proteinFile']} {cleaned_proteinFile}"
        do(cmd)

        ligands = pd.read_csv(args['ligandFile'])
        ligands['ligand'] = ligands['Smiles'].copy()
        assert 'ligand' in ligands.columns
        ligands['protein_path'] = cleaned_proteinFile
        ligands.to_csv(ligandFile_with_protein_path, index=0)

    
    header = args['header']

    if args['paper']:
        model_workdir = f"{script_folder}/workdir/big_score_model_sanyueqi_with_time"
        ckpt = "ema_inference_epoch314_model.pt"
    else:
        if args['model'] == 1:
            model_workdir = f"{script_folder}/workdir/big_score_model_sanyueqi_with_time"
            ckpt = "pro_ema_inference_epoch138_model.pt"

    if not args['rigid_protein']:
        protein_dynamic = "--protein_dynamic"
    else:
        protein_dynamic = ""
    

    if args['hts']:
        t0 = time.time()
        os.system("mkdir -p data")
        cmd = f"{python} {script_folder}/datasets/esm_embedding_preparation.py --protein_ligand_csv {ligandFile_with_protein_path} --out_file data/prepared_for_esm_{header}.fasta"
        do(cmd)
        cmd = f"MKL_SERVICE_FORCE_INTEL=1 MKL_THREADING_LAYER=GNU CUDA_VISIBLE_DEVICES={args['device']} {python} {script_folder}/esm/scripts/extract.py esm2_t33_650M_UR50D data/prepared_for_esm_{header}.fasta data/esm2_output --repr_layers 33 --include per_tok --truncation_seq_length 10000 --model_dir {script_folder}/esm_models"
        do(cmd)
        cmd = f"MKL_SERVICE_FORCE_INTEL=1 MKL_THREADING_LAYER=GNU CUDA_VISIBLE_DEVICES={args['device']} {python} {script_folder}/normal_screening.py --seed {args['seed']} --ckpt {ckpt} {protein_dynamic}"
        cmd += f" --save_visualisation --model_dir {model_workdir}  --protein_ligand_csv {ligandFile_with_protein_path} "
        cmd += f" --esm_embeddings_path data/esm2_output --out_dir {args['results']}/{header} --inference_steps {args['inference_steps']} --samples_per_complex {args['samples_per_complex']} --savings_per_complex {args['savings_per_complex']} --batch_size {args['batch_size']} --actual_steps {args['inference_steps']} --no_final_step_noise"
        do(cmd)
        t1 = time.time()
        print(t1 - t0)
    else:
        if not args['no_inference']:
            os.system("mkdir -p data")
            cmd = f"{python} {script_folder}/datasets/esm_embedding_preparation.py --protein_ligand_csv {ligandFile_with_protein_path} --out_file data/prepared_for_esm_{header}.fasta"
            do(cmd)
            cmd = f"CUDA_VISIBLE_DEVICES={args['device']} {python} {script_folder}/esm/scripts/extract.py esm2_t33_650M_UR50D data/prepared_for_esm_{header}.fasta data/esm2_output --repr_layers 33 --include per_tok --truncation_seq_length 10000 --model_dir {script_folder}/esm_models"
            do(cmd)
            cmd = f"CUDA_VISIBLE_DEVICES={args['device']} {python} {script_folder}/inference.py --seed {args['seed']} --ckpt {ckpt} {protein_dynamic}"
            cmd += f" --save_visualisation --model_dir {model_workdir}  --protein_ligand_csv {ligandFile_with_protein_path} "
            cmd += f" --esm_embeddings_path data/esm2_output --out_dir {args['results']}/{header} --inference_steps {args['inference_steps']} --samples_per_complex {args['samples_per_complex']} --savings_per_complex {args['savings_per_complex']} --batch_size {args['batch_size']} --actual_steps {args['inference_steps']} --no_final_step_noise"
            do(cmd)
            print("inference complete.")

        if not args['no_relax']:
            cmd = f"CUDA_VISIBLE_DEVICES={args['device']} {relax_python} {script_folder}/relax_final.py --results_path {args['results']}/{header} --samples_per_complex {args['samples_per_complex']} --num_workers {args['num_workers']}"
            # print("relax final step structure.")
            # exit()
            do(cmd)
            print("final step structure relax complete.")
        
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True) 
    affinity_pred = result.stdout
    print(affinity_pred)
    return float(affinity_pred)

  #      if args.movie:
  #          for i in range(len(ligands)):
   #             cmd = f"CUDA_VISIBLE_DEVICES={args.device} {relax_python} {script_folder}/movie_generation.py {args.results}/{header}/index{i}_idx_{i} 1 --python {python} --relax_python {relax_python} --inference_steps {args.inference_steps}"
     #           do(cmd)
    #            print(cmd)

#if __name__ == "__main__":
 #   args_ = parser.parse_args() 
  #  affinity_results = run_inference(args_)
