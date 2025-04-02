#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import os
import pandas as pd
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from env.dynamicbind import DBAffinity
import argparse
import time
import json
import multiprocessing
import concurrent.futures

def dump2json(obj, path):
    with open(path, 'wt') as f:
        json.dump(obj, f, indent=4)

def int2str(number, length=3):
    assert isinstance(number, int) and number < 10 ** length
    return str(number).zfill(length)

def get_db_config(args):
    return {
        'device': args.db_device,
        'batch_size': args.db_batch_size,
        'inference_steps': args.db_inference_steps,
        'script': args.db_script,
        'python': args.db_env,
        'relax_python': args.db_relax_env,
        'protein_pdb': args.db_protein_pdb,
        'results': args.db_results,
        'header': args.db_header,
        'paper': args.paper,
        'seed': args.seed,
        'rigid_protein': args.rigid_protein,
        'esm2_output': args.esm2_output,
        'savings_per_complex': args.savings_per_complex,
        'protein_ligand_csv': args.protein_ligand_csv,
        'hts': args.hts,
        'output_file': args.output_file,  
        'input_file': args.input_file, 
    }

def compute_DBAffinity_for_batch(smiles_batch, config):
    db_affinity = DBAffinity(config)
    affinity_scores = db_affinity(smiles_batch)
    return affinity_scores

def process_small_batch(smiles_batch, config):
    
    return compute_DBAffinity_for_batch(smiles_batch, config)

def compute_DBAffinity_from_dataframe(config, epoch, df):
    smiles_list = df.smiles.tolist()
    batch_size = 100 
    sub_batch_size = 5

    all_metrics = {}
    csv_counter = 0
    csv_part = 1

    for i in range(0, len(smiles_list), batch_size):
        batch = smiles_list[i:i + batch_size]
        data = []

        
        with concurrent.futures.ProcessPoolExecutor() as executor:
            sub_batches = [batch[j:j + sub_batch_size] for j in range(0, len(batch), sub_batch_size)]
            
            
            futures = [executor.submit(process_small_batch, sub_batch, config) for sub_batch in sub_batches]

            
            affinity_scores = []
            for future in concurrent.futures.as_completed(futures):
                affinity_scores.extend(future.result())

        
        for smile, score in zip(batch, affinity_scores):
            all_metrics[smile] = score
            data.append({'SMILES': smile, 'Affinity': score})
            csv_counter += 1

        
        csv_path = os.path.join(f"{config['result']}, f"{config['header']}_part_{csv_part}_100mol.csv")
        pd.DataFrame(data).to_csv(csv_path, index=False)
        print(f"Saved part {csv_part} ({csv_counter} molecules) to: {csv_path}")
        csv_part += 1

    suffix = int2str(epoch)
    output_path = os.path.join(f"{config['result']}, f"{config['header']}_{suffix}.json")
    dump2json(all_metrics, output_path)
    print(f"DBAffinity metrics saved to: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DBAffinity Calculator')
    parser.add_argument('--input_file', required=True, type=str, help='Path to input CSV file with SMILES')
    parser.add_argument('--output_file', required=True, type=str, help='Path to output CSV file')
    parser.add_argument('--protein_ligand_csv')
    parser.add_argument('--savings_per_complex', type=int, default=1)
    parser.add_argument('--db_device', type=int, default=1)
    parser.add_argument('--db_env', type=str)
    parser.add_argument('--db_relax_env', type=str)
    parser.add_argument('--db_script', type=str)
    parser.add_argument('--db_batch_size', type=int, default=8) # Размер батча теперь не важен
    parser.add_argument('--db_inference_steps', type=int, default=20)
    parser.add_argument('--db_protein_pdb', type=str)
    parser.add_argument('--db_results', type=str)
    parser.add_argument('--db_header', type=str)
    parser.add_argument('--esm2_output', type=str)
    parser.add_argument('--hts', type=str)
    parser.add_argument('-p', '--paper', action='store_true', default=False)
    parser.add_argument('--rigid_protein', action='store_true', default=False)
    parser.add_argument('--seed', help='RNG seed', type=int, default=42)
    args = parser.parse_args()

    df = pd.read_csv(args.input_file)
    config = get_db_config(args)
    epoch = 1 
    
    compute_DBAffinity_from_dataframe(config, epoch, df)
