import argparse
import os
import json
from rdkit import Chem
from freedpp.env.docking import DockingVina
from freedpp.utils import lmap


def str2strs(s):
    return s.split(',')


def str2floats(s):
    return lmap(float, s.split(','))

    
def str2ints(s):
    return lmap(int, s.split(','))


def str2bool(s):
    if s == 'True':
        return True
    elif s == 'False':
        return False
    else:
        return bool(s)


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--name', type=str, required=True)
    parser.add_argument('--exp_root', type=str, default='../experiments')
    parser.add_argument('--commands', type=str2strs, default='train,sample')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--seed', help='RNG seed', type=int, default=666)

    # Environment
    parser.add_argument('--fragments', required=True)
    parser.add_argument('--fragmentation', type=str, default='crem', choices=['crem', 'brics'])
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--starting_smile', type=str, default='c1([*:1])c([*:2])ccc([*:3])c1')
    parser.add_argument('--timelimit', type=int, default=4)

    # model updating
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--actor_lr', type=float, default=1e-4)
    parser.add_argument('--critic_lr', type=float, default=1e-4)
    parser.add_argument('--alpha_lr', type=float, default=5e-4)
    parser.add_argument('--prioritizer_lr', type=float, default=1e-4)
    parser.add_argument('--alpha_eps', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--update_num', type=int, default=256)

    # Saving and Loading
    parser.add_argument('--save_freq', type=int, default=20)
    parser.add_argument('--checkpoint', type=str, default='')

    # SAC
    parser.add_argument('--target_entropy', type=float, default=3.0)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--replay_size', type=int, default=int(1e6))
    parser.add_argument('--polyak', type=float, default=0.995)
    parser.add_argument('--tau', type=float, default=1e-1)
    parser.add_argument('--steps_per_epoch', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--train_alpha', type=str2bool, default=True)

    # Objectives and Rewards
    parser.add_argument('--objectives', type=str2strs, default=['DockingScore'])
    parser.add_argument('--weights', type=str2floats, default=[1.0])
    parser.add_argument('--reward_version', default='hard', choices=['soft', 'hard'])
    parser.add_argument('--alert_collections', type=str)

    # Sample
    parser.add_argument('--num_mols', type=int, default=1000)

    # DynamicBind
    parser.add_argument('--proteinFile', type=str, default='test.pdb')#, help='protein file')
  # parser.add_argument('--ligandFile', type=str, default='ligand.csv')#, help='contians the smiles, should contain a column named ligand')
    parser.add_argument('--samples_per_complex', type=int, default=10)#, help='num of samples data generated.')
  # parser.add_argument('--batch_size', type=int, default=32)#, help='batch size.')
    parser.add_argument('--savings_per_complex', type=int, default=10)#, help='num of samples data saved for movie generation.')
    parser.add_argument('--inference_steps', type=int, default=20)#, help='num of coordinate updates. (movie frames)')
    parser.add_argument('--header', type=str, default='test')#, help='informative name used to name result folder')
    parser.add_argument('--results', type=str, default='results')#, help='result folder.')
  # parser.add_argument('--device', type=int, default=0)#, help='CUDA_VISIBLE_DEVICES')
    parser.add_argument('--no_inference', action='store_true', default=False)#, help='used, when the inference part is already done.')
    parser.add_argument('--no_relax', action='store_true', default=False)#, help='by default, the last frame will be relaxed.')
    parser.add_argument('--movie', action='store_true', default=False)#, help='by default, no movie will generated.')
    parser.add_argument('--python', type=str, default='/home/zhangjx/anaconda3/envs/dynamicbind/bin/python')#, help='point to the python in dynamicbind env.')
    parser.add_argument('--relax_python', type=str, default='/home/zhangjx/anaconda3/envs/relax/bin/python')#, help='point to the python in relax env.')
  # parser.add_argument('-l', '--protein_path_in_ligandFile', action='store_true', default=False)#, help='read the protein from the protein_path in ligandFile.')
    parser.add_argument('--no_clean', action='store_true', default=False)#, help='by default, the input protein file will be cleaned. only take effect, when protein_path_in_ligandFile is true')
  # parser.add_argument('-s', '--ligand_is_sdf', action='store_true', default=False)#, help='ligand file is in sdf format.')
    parser.add_argument('--num_workers', type=int, default=20)#, help='Number of workers for relaxing final step structure')
    parser.add_argument('-p', '--paper', action='store_true', default=False)#, help='use paper version model.')
    parser.add_argument('--model', type=int, default=1)#, help='default model version')
  # parser.add_argument('--seed', type=int, default=42)#, help='set seed number')
    parser.add_argument('--rigid_protein', action='store_true', default=False)#, help='Use no noise in the final step of the reverse diffusion')
    parser.add_argument('--hts', action='store_true', default=False)#, help='high-throughput mode')


    # Docking
    parser.add_argument('--receptor_1', required=False)
    parser.add_argument('--box_center_1', required=False, type=str2floats)
    parser.add_argument('--box_size_1', required=False, type=str2floats)
    parser.add_argument('--receptor_2', required=False)
    parser.add_argument('--box_center_2', required=False, type=str2floats)
    parser.add_argument('--box_size_2', required=False, type=str2floats)
    parser.add_argument('--vina_program', required=False)
    parser.add_argument('--exhaustiveness', type=int, default=8)
    parser.add_argument('--num_modes', type=int, default=10)
    parser.add_argument('--num_sub_proc', type=int, default=1)
    parser.add_argument('--n_conf', type=int, default=3)
    parser.add_argument('--error_val', type=float, default=99.9)
    parser.add_argument('--timeout_gen3d', type=int, default=None)
    parser.add_argument('--timeout_dock', type=int, default=None)

    # Metrics
    parser.add_argument('--unique_k', type=str2ints, default=[1000, 1000])
    parser.add_argument('--n_jobs', type=int, default=1)

    # Actor and Critic Architectures
    parser.add_argument('--n_nets', type=int, default=2)
    parser.add_argument('--merger', type=str, default='ai', choices=['mi', 'ai'])
    parser.add_argument('--action_mechanism', type=str, default='pi', choices=['sfps', 'pi'])
    parser.add_argument('--ecfp_size', type=int, default=1024)
    parser.add_argument('--emb_size', type=int, default=128)
    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--aggregation', type=str, default='sum', choices=['sum', 'mean'])

    # PER-PE
    parser.add_argument('--per', type=str2bool, default=False)
    parser.add_argument('--dzeta', type=float, default=0.6)
    parser.add_argument('--beta_start', type=float, default=0.4)
    parser.add_argument('--beta_frames', type=float, default=100000)

    return parser.parse_args()


def update_args(args):
    args.exp_dir = os.path.join(args.exp_root, args.name)
    #args.difference = get_docking(args)
    args.db_config = get_db_config(args)
    args.docking_config_1 = get_docking_config_1(args)
    args.docking_config_2 = get_docking_config_2(args)

    args.mols_dir = os.path.join(args.exp_dir, 'mols')
    args.model_dir = os.path.join(args.exp_dir, 'ckpt')
    args.logs_dir = os.path.join(args.exp_dir, 'logs')
    args.metrics_dir = os.path.join(args.exp_dir, 'metrics')

    args.atom_vocab = get_atom_vocab()
    args.bond_vocab = get_bond_vocab()
    with open(os.path.join(args.fragments)) as f:
        args.frag_vocab = json.load(f)

def get_db_config(args):
    db_config = {
        'proteinFile': args.proteinFile,
       #'ligandFile': args.ligandFile,
        'samples_per_complex': args.samples_per_complex,
        'batch_size': args.batch_size,
        'savings_per_complex': args.savings_per_complex,
        'inference_steps': args.inference_steps,
        'header': args.header,
        'results': args.results,
        'device': args.device,
        'no_inference': args.no_inference,
        'no_relax': args.no_relax,
       #'movie': args.movie,
        'python': args.python,
        'relax_python': args.relax_python,
       #'protein_path_in_ligandFile': args.protein_path_in_ligandFile,
        'no_clean': args.no_clean,
       #'ligand_is_sdf': args.ligand_is_sdf,
        'num_workers': args.num_workers,
        'paper': args.paper,
        'model': args.model,
        'seed': args.seed,
        'rigid_protein': args.rigid_protein,
        'hts': args.hts
    }

    return db_config


def get_docking_config_1(args):
    docking_config_1 = {
        'receptor': args.receptor_1,
        'box_center': args.box_center_1,
        'box_size': args.box_size_1,
        'vina_program': args.vina_program,
        'exhaustiveness': args.exhaustiveness,
        'num_sub_proc': args.num_sub_proc,
        'num_modes': args.num_modes,
        'timeout_gen3d': args.timeout_gen3d,
        'timeout_dock': args.timeout_dock,
        'seed': args.seed,
        'n_conf': args.n_conf,
        'error_val': args.error_val
    }

    return docking_config_1

def get_docking_config_2(args):
    docking_config_2 = {
        'receptor': args.receptor_2,
        'box_center': args.box_center_2,
        'box_size': args.box_size_2,
        'vina_program': args.vina_program,
        'exhaustiveness': args.exhaustiveness,
        'num_sub_proc': args.num_sub_proc,
        'num_modes': args.num_modes,
        'timeout_gen3d': args.timeout_gen3d,
        'timeout_dock': args.timeout_dock,
        'seed': args.seed,
        'n_conf': args.n_conf,
        'error_val': args.error_val
    }

    return docking_config_2

def get_atom_vocab():
    atom_vocab = ['H', 'C', 'N', 'O', 'S', 'P', 'F', 'I', 'Cl', 'Br']
    return atom_vocab


def get_bond_vocab():
    bond_vocab = [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ]
    return bond_vocab
