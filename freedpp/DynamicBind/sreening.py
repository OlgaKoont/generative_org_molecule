import copy
import os
import torch
import shutil
import warnings
warnings.filterwarnings("ignore")

import time
from argparse import ArgumentParser, Namespace, FileType
from rdkit.Chem import RemoveHs
from functools import partial
import numpy as np
import pandas as pd
import scipy
from Bio.PDB import PDBParser

from rdkit import RDLogger
from rdkit.Chem import MolFromSmiles, AddHs
from rdkit import Chem

import torch
torch.set_num_threads(1)
torch.multiprocessing.set_sharing_strategy('file_system')



from torch_geometric.loader import DataLoader


from datasets.process_mols import read_molecule, generate_conformer, write_mol_with_coords
from datasets.pdbbind import PDBBind,PDBBindScoring
from utils.diffusion_utils import t_to_sigma as t_to_sigma_compl, get_t_schedule, set_time
from utils.sampling import randomize_position, sampling
from utils.utils import get_model
from utils.visualise import LigandToPDB, modify_pdb, receptor_to_pdb, save_protein
from utils.clash import compute_side_chain_metrics
# from utils.relax import openmm_relax
from tqdm import tqdm
import datetime
from contextlib import contextmanager

from multiprocessing import Pool as ThreadPool

import random
import pickle
# pool = ThreadPool(8)

@contextmanager
def Timer(title):
    'timing function'
    t0 = datetime.datetime.now()
    yield
    print("%s - done in %is"%(title, (datetime.datetime.now() - t0).seconds))
    return None

RDLogger.DisableLog('rdApp.*')
import yaml

def screen(args):

    args = parser.parse_args()
    def Seed_everything(seed=42):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    Seed_everything(seed=args.seed)
    if args.config:
        config_dict = yaml.load(args.config, Loader=yaml.FullLoader)
        arg_dict = args.__dict__
        for key, value in config_dict.items():
            if isinstance(value, list):
                for v in value:
                    arg_dict[key].append(v)
            else:
                arg_dict[key] = value

    os.makedirs(args.out_dir, exist_ok=True)

    with open(f'{args.model_dir}/model_parameters.yml') as f:
        score_model_args = Namespace(**yaml.full_load(f))

    if args.confidence_model_dir is not None:
        with open(f'{args.confidence_model_dir}/model_parameters.yml') as f:
            confidence_args = Namespace(**yaml.full_load(f))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.protein_ligand_csv is not None:
        df = pd.read_csv(args.protein_ligand_csv)
        # df = df[:10]
        if 'crystal_protein_path' not in df.columns:
            df['crystal_protein_path'] = df['protein_path']
        protein_path_list = df['protein_path'].tolist()
        ligand_descriptions = df['ligand'].tolist()
        df['name'] = [f'idx_{i}' for i in range(df.shape[0])]
        name_list = df['name'].tolist()
    else:
        protein_path_list = [args.protein_path]
        ligand_descriptions = [args.ligand]

    test_dataset = PDBBindScoring(transform=None, root='', name_list=name_list, protein_path_list=protein_path_list, ligand_descriptions=ligand_descriptions,
                           receptor_radius=score_model_args.receptor_radius, cache_path=args.cache_path,
                           remove_hs=score_model_args.remove_hs, max_lig_size=None,
                           c_alpha_max_neighbors=score_model_args.c_alpha_max_neighbors, matching=False, keep_original=False,
                           popsize=score_model_args.matching_popsize, maxiter=score_model_args.matching_maxiter,center_ligand=True,
                           all_atoms=score_model_args.all_atoms, atom_radius=score_model_args.atom_radius,
                           atom_max_neighbors=score_model_args.atom_max_neighbors,
                           esm_embeddings_path= args.esm_embeddings_path if score_model_args.esm_embeddings_path is not None else None,
                           require_ligand=True,require_receptor=True, num_workers=args.num_workers, keep_local_structures=args.keep_local_structures, use_existing_cache=args.use_existing_cache)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

    t_to_sigma = partial(t_to_sigma_compl, args=score_model_args)

    model = get_model(score_model_args, device, t_to_sigma=t_to_sigma, no_parallel=True)
    state_dict = torch.load(f'{args.model_dir}/{args.ckpt}', map_location=torch.device('cpu'))
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()

    if args.confidence_model_dir is not None:
        if confidence_args.transfer_weights:
            with open(f'{confidence_args.original_model_dir}/model_parameters.yml') as f:
                confidence_model_args = Namespace(**yaml.full_load(f))
        else:
            confidence_model_args = confidence_args
        confidence_model = get_model(confidence_model_args, device, t_to_sigma=t_to_sigma, no_parallel=True, confidence_mode=True)
        state_dict = torch.load(f'{args.confidence_model_dir}/{args.confidence_ckpt}', map_location=torch.device('cpu'))
        confidence_model.load_state_dict(state_dict, strict=True)
        confidence_model = confidence_model.to(device)
        confidence_model.eval()
    else:
        confidence_model = None
        confidence_args = None
        confidence_model_args = None

    tr_schedule = get_t_schedule(inference_steps=args.inference_steps)
    rot_schedule = tr_schedule
    tor_schedule = tr_schedule
    res_tr_schedule = tr_schedule
    res_rot_schedule = tr_schedule
    res_chi_schedule = tr_schedule
    print('common t schedule', tr_schedule)

    failures, skipped, confidences_list, names_list, run_times, min_self_distances_list = 0, 0, [], [], [], []
    N = args.samples_per_complex
    print('Size of test dataset: ', len(test_dataset))

    affinity_pred = {}
    all_complete_affinity = []
    steps = args.actual_steps if args.actual_steps is not None else args.inference_steps
    # data_list = list(chain(*[[copy.deepcopy(orig_complex_graph) for _ in range(N)] for orig_complex_graph in test_loader]))
    data_list = []
    complex_names = []
    for orig_complex_graph in test_loader:
        complex_names.append(orig_complex_graph.name[0])
        for _ in range(N):
            data_list.append(copy.deepcopy(orig_complex_graph))


    randomize_position(
        data_list,
        score_model_args.no_torsion,
        args.no_random,
        score_model_args.tr_sigma_max,
        score_model_args.rot_sigma_max,
        score_model_args.tor_sigma_max,
        score_model_args.res_tr_sigma_max,
        score_model_args.res_rot_sigma_max
    )

    all_lddt_pred, all_affinity_pred = [],[]
    I = int(np.ceil(len(data_list) / args.batch_size))
    for i in range(I):
        batch = data_list[i*args.batch_size:(i+1)*args.batch_size]
        # complexes_names = [data.name[0] for data in batch]
        outputs = sampling(
            data_list=batch,
            model=model,
            inference_steps=steps,
            tr_schedule=tr_schedule,
            rot_schedule=rot_schedule,
            tor_schedule=tor_schedule,
            res_tr_schedule=res_tr_schedule,
            res_rot_schedule=res_rot_schedule,
            res_chi_schedule=res_chi_schedule,
            device=device,
            t_to_sigma=t_to_sigma,
            model_args=score_model_args,
            no_random=args.no_random,
            ode=args.ode,
            visualization_list=None,
            batch_size=args.batch_size,
            no_final_step_noise=args.no_final_step_noise,
            protein_dynamic=args.protein_dynamic
        )
        all_lddt_pred.append(outputs[2])
        all_affinity_pred.append(outputs[3])

    all_lddt_pred = torch.cat(all_lddt_pred).cpu().split(N)
    all_affinity_pred = torch.cat(all_affinity_pred).cpu().split(N)
    for complex_name, lddt_pred, affinity_pred_ in zip(complex_names, all_lddt_pred, all_affinity_pred):
        lddt_pred = lddt_pred.view(-1).numpy()
        affinity_pred_ = affinity_pred_.view(-1).numpy()
        final_affinity_pred = np.minimum((affinity_pred_ * lddt_pred).sum() / (lddt_pred.sum() + 1e-12), 15.)

        affinity_pred[complex_name] = final_affinity_pred

        complete_affinity = pd.DataFrame(
            {
                'name':complex_name,
                'lddt':lddt_pred,
                'affinity':affinity_pred_
            }
        )
        all_complete_affinity.append(complete_affinity)

    print(f'Failed for {failures} complexes')
    print(f'Skipped {skipped} complexes')

    affinity_pred_df = pd.DataFrame(
        {
            'name':list(affinity_pred.keys()),
            'affinity':list(affinity_pred.values())
        }
    )
    affinity_pred_df.to_csv(f'{args.out_dir}/affinity_prediction.csv',index=False)
    pd.concat(all_complete_affinity).to_csv(f'{args.out_dir}/complete_affinity_prediction.csv',index=False)

    x = affinity_pred_df['affinity']

    print(f'Results are in {args.out_dir}')
    print(x)
    return float(x)

args = parser.parse_args()
affinity_results = screen(args)
print(x)
