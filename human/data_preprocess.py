from rdkit import Chem
import rdkit.Chem
import torch_geometric
import torch_cluster
from constants import ATOM_VOCAB
from pdb_graph import _rbf, _normalize
from Bio.PDB import PDBParser, NeighborSearch
import numpy as np
import pandas as pd
import torch
from torch.utils import data
import sys
import os
import dgl
import deepchem
from subword_nmt.apply_bpe import BPE
import codecs
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
import networkx as nx

pk = deepchem.dock.ConvexHullPocketFinder()
sys.path.append('..')
os.chdir(os.path.dirname(os.path.abspath(__file__)))
num_atom_feat = 75
res_atom_feat = 35

seq_voc = "ABCDEFGHIKLMNOPQRSTUVWXYZ"
seq_dict = {v: (i + 1) for i, v in enumerate(seq_voc)}
seq_dict_len = len(seq_dict)
max_seq_len = 545
pdb_cache = {}
token_dict = {
    '_PAD': 0, '_GO': 1, '_EOS': 2, '_UNK': 3, 'ALA': 4, 'ARG': 5, 'ASN': 6,
    'ASP': 7, 'CYS': 8, 'GLY': 9, 'GLU': 10, 'GLY': 11, 'HIS': 12, 'ILE': 13, 'LEU': 14,
    'LYS': 15, 'MET': 16, 'PHE': 17, 'PRO': 18, 'SER': 19, 'THR': 20, 'TRP': 21,
    'TYR': 22, 'VAL': 23, 'X': 24, 'U': 25, 'O': 26, 'B': 27, 'Z': 28
}

def extract_c_alpha_atoms(pdb_file):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_file)
    ca_atoms = []
    residues = []

    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.has_id('CA'):
                    ca_atoms.append(residue['CA'])
                    residues.append(residue.get_resname())

    return ca_atoms, residues


def one_hot_encode(residues, token_dict):
    one_hot_features = []
    for res in residues:
        one_hot = np.zeros(len(token_dict))
        if res in token_dict:
            one_hot[token_dict[res]] = 1
        else:
            one_hot[token_dict['_UNK']] = 1
        one_hot_features.append(one_hot)

    return np.array(one_hot_features)


def construct_adjacency_matrix(ca_atoms, threshold=8.0):
    num_residues = len(ca_atoms)
    adjacency_matrix = np.zeros((num_residues, num_residues))
    ns = NeighborSearch(ca_atoms)

    for i, ca_atom in enumerate(ca_atoms):
        neighbors = ns.search(ca_atom.coord, threshold)
        for neighbor in neighbors:
            j_index = ca_atoms.index(neighbor)
            adjacency_matrix[i, j_index] = 1

    return adjacency_matrix


def onehot_encoder(a=None, alphabet=None, default=None, drop_first=False):
    alphabet_set = set(alphabet)
    a = [x if x in alphabet_set else default for x in a]
    a = pd.Categorical(a, categories=alphabet)
    onehot = pd.get_dummies(pd.Series(a), columns=alphabet, drop_first=drop_first)
    return onehot.values


def _build_atom_feature(mol):
    feature_alphabet = {
        'GetSymbol': (ATOM_VOCAB, 'unk'),
        'GetDegree': ([0, 1, 2, 3, 4, 5, 6], 6),
        'GetTotalNumHs': ([0, 1, 2, 3, 4, 5, 6], 6),
        'GetImplicitValence': ([0, 1, 2, 3, 4, 5, 6], 6),
        'GetIsAromatic': ([0, 1], 1)
    }

    atom_feature = None
    for attr in ['GetSymbol', 'GetDegree', 'GetTotalNumHs',
                 'GetImplicitValence', 'GetIsAromatic']:
        feature = [getattr(atom, attr)() for atom in mol.GetAtoms()]
        feature = onehot_encoder(feature,
                                 alphabet=feature_alphabet[attr][0],
                                 default=feature_alphabet[attr][1],
                                 drop_first=(attr in ['GetIsAromatic'])
                                 )
        atom_feature = feature if atom_feature is None else np.concatenate((atom_feature, feature), axis=1)
    atom_feature = atom_feature.astype(np.float32)

    return atom_feature


def _build_edge_feature(coords, edge_index, D_max=4.5, num_rbf=16):
    E_vectors = coords[edge_index[0]] - coords[edge_index[1]]
    rbf = _rbf(E_vectors.norm(dim=-1), D_max=D_max, D_count=num_rbf)
    edge_s = rbf
    edge_v = _normalize(E_vectors).unsqueeze(-2)
    edge_s, edge_v = map(torch.nan_to_num, (edge_s, edge_v))

    return edge_s, edge_v


def sdf_to_graphs_list(file_path):
    graphs = []
    supplier = rdkit.Chem.SDMolSupplier(file_path, sanitize=False)
    for mol in supplier:
        graphs.append(featurize_drug(mol, model='no-trans'))
    return graphs


def featurize_drug(sdf_path, name=None, edge_cutoff=4.5, num_rbf=16, model='trans'):
    if model == 'trans':
        mol = rdkit.Chem.MolFromMolFile(sdf_path)
    else:
        mol = sdf_path
    conf = mol.GetConformer()
    with torch.no_grad():
        coords = conf.GetPositions()
        coords = torch.as_tensor(coords, dtype=torch.float32)
        atom_feature = _build_atom_feature(mol)
        atom_feature = torch.as_tensor(atom_feature, dtype=torch.float32)
        edge_index = torch_cluster.radius_graph(coords, r=edge_cutoff)

    node_s = atom_feature
    node_v = coords.unsqueeze(1)
    edge_s, edge_v = _build_edge_feature(
        coords, edge_index, D_max=edge_cutoff, num_rbf=num_rbf)

    data = torch_geometric.data.Data(
        x=coords, edge_index=edge_index, name=name,
        node_v=node_v, node_s=node_s, edge_v=edge_v, edge_s=edge_s)

    return data

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input.csv {0} not in allowable set{1}:".format(
            x, allowable_set))
    return [x == s for s in allowable_set]


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]


def atom_features(atom,
                  explicit_H=False,
                  use_chirality=False):
    results = one_of_k_encoding_unk(
        atom.GetSymbol(),
        [
            'C',
            'N',
            'O',
            'S',
            'F',
            'Si',
            'P',
            'Cl',
            'Br',
            'Mg',
            'Na',
            'Ca',
            'Fe',
            'As',
            'Al',
            'I',
            'B',
            'V',
            'K',
            'Tl',
            'Yb',
            'Sb',
            'Sn',
            'Ag',
            'Pd',
            'Co',
            'Se',
            'Ti',
            'Zn',
            'H',
            'Li',
            'Ge',
            'Cu',
            'Au',
            'Ni',
            'Cd',
            'In',
            'Mn',
            'Zr',
            'Cr',
            'Pt',
            'Hg',
            'Pb',
            'other'
        ]) + one_of_k_encoding(atom.GetDegree(),
                               [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) + \
              one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6]) + \
              [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
              one_of_k_encoding_unk(atom.GetHybridization(), [
                  Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                  Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.
                                    SP3D, Chem.rdchem.HybridizationType.SP3D2
              ]) + [atom.GetIsAromatic()]
    if not explicit_H:
        results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),
                                                  [0, 1, 2, 3, 4])
    if use_chirality:
        try:
            results = results + one_of_k_encoding_unk(
                atom.GetProp('_CIPCode'),
                ['R', 'S']) + [atom.HasProp('_ChiralityPossible')]
        except:
            results = results + [False, False
                                 ] + [atom.HasProp('_ChiralityPossible')]

    return results


def atom_feature2(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'P', 'Cl', 'Br', 'B', 'H']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5]) +
                    [atom.GetIsAromatic()])  # total 31


def get_atom_feature(m):
    H = []
    for i in range(len(m)):
        H.append(atom_feature2(m[i][0]))
    H = np.array(H)

    return H


def process_protein(pdb_file):
    m = Chem.MolFromPDBFile(pdb_file, sanitize=False)
    if m is None:
        raise ValueError(f"Could not parse PDB file {pdb_file}")
    am = GetAdjacencyMatrix(m)
    pockets = pk.find_pockets(pdb_file)
    n2 = m.GetNumAtoms()
    c2 = m.GetConformers()[0]
    d2 = np.array(c2.GetPositions())
    binding_parts = []
    not_in_binding = [i for i in range(0, n2)]
    constructed_graphs = []
    for bound_box in pockets:
        x_min = bound_box.x_range[0]
        x_max = bound_box.x_range[1]
        y_min = bound_box.y_range[0]
        y_max = bound_box.y_range[1]
        z_min = bound_box.z_range[0]
        z_max = bound_box.z_range[1]
        binding_parts_atoms = []
        idxs = []
        for idx, atom_cord in enumerate(d2):
            if x_min < atom_cord[0] < x_max and y_min < atom_cord[1] < y_max and z_min < atom_cord[2] < z_max:
                binding_parts_atoms.append((m.GetAtoms()[idx], atom_cord))
                idxs.append(idx)
                if idx in not_in_binding:
                    not_in_binding.remove(idx)
        ami = am[np.ix_(idxs, idxs)]
        H = get_atom_feature(binding_parts_atoms)
        g = nx.convert_matrix.from_numpy_array(ami)
        graph = dgl.from_networkx(g)
        graph.ndata['h'] = torch.Tensor(H)
        graph = dgl.add_self_loop(graph)
        constructed_graphs.append(graph)
        binding_parts.append(binding_parts_atoms)

    num = len(constructed_graphs)
    constructed_graphs = dgl.batch(constructed_graphs)

    return binding_parts, not_in_binding, constructed_graphs, num


def adjacent_matrix(mol):
    adjacency = Chem.GetAdjacencyMatrix(mol)
    return np.array(adjacency) + np.eye(adjacency.shape[0])


class Data_Encoder(data.Dataset):
    def __init__(self, txtpath, sdf_directory, sdf_map_path, pdb_directory, pdb_map_path):
        self.mol_graphs = {}
        self.p_graphs = {}
        self.p_num_dic = {}
        self.p_num = {}
        self.p_list = []
        self.pdb_id_list = []
        self.pv_dic = {}
        self.input_p_mask_dic = {}
        self.p_graphs_dic ={}
        vocab_path = '/home/XuLe/ESPF/protein_codes_uniprot.txt'
        bpe_codes_protein = codecs.open(vocab_path)
        self.pbpe = BPE(bpe_codes_protein, merges=-1, separator='')
        sub_csv = pd.read_csv('/home/XuLe/ESPF/subword_units_map_uniprot.csv')

        idx2word_p = sub_csv['index'].values
        self.words2idx_p = dict(zip(idx2word_p, range(0, len(idx2word_p))))

        vocab_path = '/home/XuLe/ESPF/drug_codes_chembl.txt'
        bpe_codes_drug = codecs.open(vocab_path)
        self.dbpe = BPE(bpe_codes_drug, merges=-1, separator='')
        sub_csv = pd.read_csv('/home/XuLe/ESPF/subword_units_map_chembl.csv')
        idx2word_d = sub_csv['index'].values

        self.words2idx_d = dict(zip(idx2word_d, range(0, len(idx2word_d))))
        with open(txtpath, "r") as f:
            data_list = f.read().strip().split('\n')
        smiles, sequences, interactions = [], [], []
        for no, data in enumerate(data_list):
            smile, sequence, interaction = data.strip().split()
            smiles.append(smile)
            sequences.append(sequence)
            interactions.append(interaction)

        self.smiles = smiles
        self.Sequences = sequences
        self.interactions = interactions
        self.pdb_directory = sdf_directory
        self.pdb_directory = pdb_directory
        self.pdb_map_path = pdb_map_path
        with open(pdb_map_path, 'r') as f:
            lines = f.readlines()
        self.pdb_id_to_sequence = {}
        self.num = {}
        self.sdf_directory = sdf_directory
        self.sdf_map_path = sdf_map_path
        with open(sdf_map_path, 'r') as f:
            lines = f.readlines()
        self.sdf_id_to_sequence = {}
        for line in lines:
            parts = line.strip().split()

    def __len__(self):
        return len(self.interactions)

    def get_protein_graphs(self, pdb_id_full):
        pdb_id = pdb_id_full
        if pdb_id not in self.p_graphs:
            pdb_path = os.path.join(self.pdb_directory, f"{pdb_id}.pdb")
            try:
                _, _, constructed_graphs, number = process_protein(pdb_path)
                self.p_graphs[pdb_id] = constructed_graphs
                self.p_num[pdb_id] = number
            except Exception as e:
                print(f"Error processing protein {pdb_id}: {e}")
                return None, None
        else:
            constructed_graphs = self.p_graphs[pdb_id]
            number = self.p_num.get(pdb_id, -1)

        return constructed_graphs, number

    def p_to_embedding(self, prot):
        x = np.zeros(max_seq_len)
        for i, ch in enumerate(prot[:max_seq_len]):
            if ch in seq_dict:
                x[i] = seq_dict[ch]
            else:
                print(f"Warning: Unrecognized character '{ch}' ")
                x[i] = 0
        return x

    def get_mol_embedding(self, sdf_id_full):
        sdf_id = sdf_id_full[:4]
        sdf_path = os.path.join(self.sdf_directory, f"{sdf_id}.sdf")
        graphs = sdf_to_graphs_list(sdf_path)

        return graphs

    def get_protein_embedding(self, pdb_id_full):
        global pdb_cache
        pdb_id = pdb_id_full
        pdb_path = os.path.join(self.pdb_directory, f"{pdb_id}.pdb")

        if pdb_id in pdb_cache:
            return pdb_cache[pdb_id]
        ca_atoms, residues = extract_c_alpha_atoms(pdb_path)
        node_features = one_hot_encode(residues, token_dict)
        adjacency_matrix = construct_adjacency_matrix(ca_atoms)
        total_atoms = node_features.shape[0]
        pdb_cache[pdb_id] = (node_features, adjacency_matrix, total_atoms)

        return node_features, adjacency_matrix, total_atoms

    def __getitem__(self, index):
        d = self.smiles[index]
        p = self.Sequences[index]
        label = np.array(self.interactions[index], dtype=np.float32)
        d_v, input_mask_d = self.drug2emb_encoder(d)
        if p in self.p_list:
            p_v = self.pv_dic[p]
            input_mask_p = self.input_p_mask_dic[p]
        else:
            p_v, input_mask_p = self.protein2emb_encoder(p)
            self.pv_dic[p] = p_v
            self.input_p_mask_dic[p] = input_mask_p
            self.p_list.append(p)
        atom_feature, adj, num_size = self.mol_features(d)

        d_v = torch.FloatTensor(d_v)
        p_v = torch.FloatTensor(p_v)
        label = torch.LongTensor(label)
        input_mask_d = torch.LongTensor(input_mask_d)
        input_mask_p = torch.LongTensor(input_mask_p)

        atom_feature = torch.FloatTensor(atom_feature)
        adj = torch.FloatTensor(adj)
        num_size = torch.tensor(num_size)

        pdb_id = None
        with open(self.pdb_map_path, 'r', newline='') as f:
            for line in f:
                if self.Sequences[index] in line:
                    pdb_id = line.split()[3]
                    break
        if pdb_id is None:
            raise RuntimeError(f"PDB ID not found for sequence {p}")

        node_feature, p_adj, res_size = self.get_protein_embedding(pdb_id)
        node_feature = torch.FloatTensor(node_feature)
        p_adj = torch.FloatTensor(p_adj)
        res_size = torch.tensor(res_size)

        sdf_id = None
        with open(self.sdf_map_path, 'r', newline='') as f:
            for line in f:
                if self.smiles[index] in line:
                    sdf_id = line.split()[3]
                    break

        if sdf_id is None:
            raise RuntimeError(f"PDB ID not found for sequence {p}")

        graphs = self.get_mol_embedding(sdf_id)

        if pdb_id is None:
            raise RuntimeError(f"PDB ID not found for sequence {p}")

        try:
            if pdb_id in self.pdb_id_list:
                protein_graphs = self.p_graphs_dic[pdb_id]
                protein_num = self.p_num_dic[pdb_id]
            else:
                protein_graphs, protein_num = self.get_protein_graphs(pdb_id)
                self.p_graphs_dic[pdb_id] = protein_graphs
                self.p_num_dic[pdb_id] = protein_num
                self.pdb_id_list.append(pdb_id)
        except RuntimeError as e:
            print(e)

        sample = {'d_v': d_v, 'input_mask_d': input_mask_d,
                  'atom_feature': atom_feature, 'adj': adj, 'num_size': num_size,'d_graphs':graphs,
                  'p_v': p_v, 'input_mask_p': input_mask_p,
                  'p_graphs': protein_graphs, 'p_num': protein_num,
                  'node_feature': node_feature, 'p_adj': p_adj, 'res_size': res_size,
                  'label': label }

        return sample

    def drug2emb_encoder(self, x):
        max_d = 50
        t1 = self.dbpe.process_line(x).split()
        try:
            i1 = np.asarray([self.words2idx_d[i] for i in t1])
        except:
            i1 = np.array([0])
        l = len(i1)
        if l < max_d:
            i = np.pad(i1, (0, max_d - l), 'constant', constant_values=0)
            input_mask = ([1] * l) + ([0] * (max_d - l))
        else:
            i = i1[:max_d]
            input_mask = [1] * max_d
        return i, np.asarray(input_mask)

    def protein2emb_encoder(self, x):
        max_p = 545
        t1 = self.pbpe.process_line(x).split()
        try:
            i1 = np.asarray([self.words2idx_p[i] for i in t1])
        except:
            i1 = np.array([0])
        l = len(i1)
        if l < max_p:
            i = np.pad(i1, (0, max_p - l), 'constant', constant_values=0)
            input_mask = ([1] * l) + ([0] * (max_p - l))
        else:
            i = i1[:max_p]
            input_mask = [1] * max_p
        return i, np.asarray(input_mask)

    def mol_features(self, smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
        except:
            raise RuntimeError("SMILES cannot been parsed!")
        num_size = Chem.MolFromSmiles(smiles).GetNumAtoms()
        atom_feat = np.zeros((mol.GetNumAtoms(), num_atom_feat))
        for atom in mol.GetAtoms():
            atom_feat[atom.GetIdx(), :] = atom_features(atom)
        adj_matrix = adjacent_matrix(mol)
        return atom_feat, adj_matrix, num_size

