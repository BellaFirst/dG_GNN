import numpy as np
import torch
import json
import dgl
from dgl import DGLGraph
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors as rdDesc

class Smile2Graph:
    def __init__(self, compounds) -> None:
        # self.smile_list = smile_list

        self.compounds = compounds
        
        self.graph_data = dict() # important
        self.graph_dict = dict() # convert to graph_data

        self.node_features = dict()
        self.edge_features = dict()

        self.node_idx = {
            'node_num': [],
        }

        self.subgraphs = self.compounds

        self.atom_idx = 0
        self.allowable_sets = {
            "symbol": ["None", "B", "Br", "C", "Ca", "Cl", "F", "H", "I", "N", "Na", "O", "P", "S"],
            "atomic_volume": {'H': 14.4, 'Li': 13.1, 'Na': 23.7, 'K': 45.46, 'Rb': 55.9, 'Cs': 71.07, 'Be': 5.0, 'Mg': 13.9,
                        'Ca': 29.9, 'Sr': 33.7,
                        'Ba': 39.24, 'Sc': 15.0, 'Y': 19.8, 'Ti': 10.64, 'Zr': 14.1, 'Hf': 13.6, 'V': 8.78, 'Nb': 10.87,
                        'Ta': 10.9, 'Cr': 7.23,
                        'Mo': 9.4, 'W': 9.53, 'Mn': 1.39, 'Tc': 8.5, 'Re': 8.85, 'Fe': 7.1, 'Ru': 8.3, 'Os': 8.49,
                        'Co': 6.7, 'Rh': 8.3, 'Ir': 8.54,
                        'Ni': 6.59, 'Pd': 8.9, 'Pt': 9.1, 'Cu': 7.1, 'Ag': 10.3, 'Au': 10.2, 'Zn': 9.2, 'Cd': 13.1,
                        'Hg': 14.82, 'B': 4.6, 'Al': 10.0,
                        'Ga': 11.8, 'In': 15.7, 'Tl': 7.2, 'C': 4.58, 'Si': 12.1, 'Ge': 13.6, 'Sn': 16.3, 'Pb': 18.17,
                        'N': 17.3, 'P': 17.0, 'As': 13.1,
                        'Sb': 18.23, 'Bi': 21.3, 'O': 14.0, 'S': 15.5, 'Se': 16.45, 'Te': 20.5, 'Po': 22.23, 'F': 17.1,
                        'Cl': 22.7, 'Br': 23.5,
                        'I': 25.74, 'La': 20.73, 'Ce': 20.67, 'Pr': 20.8, 'Nd': 20.6, 'Pm': 28.9, 'Sm': 19.95, 'Eu': 28.9,
                        'Gd': 19.9, 'Tb': 19.2,
                        'Dy': 19.0, 'Ho': 18.7, 'Er': 18.4, 'Tm': 18.1, 'Yb': 24.79, 'Lu': 17.78, 'Ne': 16.7, 'Ar': 28.5,
                        'Kr': 38.9, 'Xe': 37.3,
                        'Rn': 50.5, 'Ra': 45.2, 'Ac': 22.54, 'Th': 19.9, 'Pa': 15, 'U': 12.59, 'Np': 11.62, 'Pu': 12.32,
                        'Am': 17.86, 'Cm': 18.28, 
                        '*': 0},
            "n_valence": [0, 1, 2, 3, 4, 5, 6],
            "n_hydrogens": [0, 1, 2, 3, 4],
            "n_radical_electrons": [0, 1],
            "degree": [0, 1, 2, 3, 4, 5, 6],
            "formal_charge": [-1, 1],
            "hybridization": ["s", "sp", "sp2", "sp3"], 
        }

        self.G = self.smile2graph()

    def get_atom_features(self, atom):

        atom_features = []

        # symbol
        atom_features.append(self.encod(atom.GetSymbol(), self.allowable_sets["symbol"]))

        # 原子体积
        atom_features.append(self.allowable_sets["atomic_volume"][atom.GetSymbol()])

        # 隐含价
        atom_features.append(self.encod(atom.GetImplicitValence(), self.allowable_sets['n_valence']))

        #  
        atom_features.append(self.encod(atom.GetTotalNumHs(), self.allowable_sets['n_hydrogens']))

        # 价电子
        atom_features.append(self.encod(atom.GetNumRadicalElectrons(), self.allowable_sets['n_radical_electrons']))

        # 成键
        atom_features.append(self.encod(atom.GetDegree(), self.allowable_sets['degree']))

        # 电荷
        atom_features.append(self.encod(atom.GetFormalCharge(), self.allowable_sets['formal_charge']))

        # 杂化类型
        atom_features.append(self.encod(atom.GetHybridization(), self.allowable_sets['hybridization']))
        
        return atom_features

    def encod(self, x, atom_list):
        '将x与atom_list逐个比较 相同则返回索引值 不同则返回0'
        if x in atom_list:
            return atom_list.index(x)
        else:
            return 0

    def get_bond_features(self, bond):
        """
        边特征，包括：是否为单键、双键、三键、成环、芳香环、共轭
        """
        bond_type = bond.GetBondType()
        bond_feats = [
            bond_type == Chem.rdchem.BondType.SINGLE, 
            bond_type == Chem.rdchem.BondType.DOUBLE,
            bond_type == Chem.rdchem.BondType.TRIPLE, 
            bond_type == Chem.rdchem.BondType.AROMATIC,
            bond.IsInRing(), 
            bond.GetIsConjugated(),
        ]
        return bond_feats

    def get_bond_type(self, bond):

        bond_type = bond.GetBondType()

        if bond_type==Chem.rdchem.BondType.SINGLE:

            bond_type = 'single'

        elif bond_type==Chem.rdchem.BondType.DOUBLE:

            bond_type = 'double'

        elif bond_type == Chem.rdchem.BondType.TRIPLE:

            bond_type = 'triple'

        elif bond_type == Chem.rdchem.BondType.AROMATIC:

            bond_type = 'aromatic'

        # elif bond.GetIsConjugated():

        #     bond_type = 'isConjugated'

        # elif bond.IsInRing():

        #     bond_type = 'isInRing'

        else:

            bond_type = 'None'

        return bond_type

    def init_u_v(self):

        uv = {
            'u':[],
            'v':[]
        }
        return uv 

    def init_subgraph(self, cpd):

        self.subgraphs[cpd]['subgraph'] = {
            'node_num': [],
        }


    def add2subgraph(self, cpd, node_num, node_type):

        if node_num not in self.subgraphs[cpd]['subgraph']['node_num']:
            
            self.subgraphs[cpd]['subgraph']['node_num'].append(node_num)

            if node_type not in self.subgraphs[cpd]['subgraph'].keys():

                self.subgraphs[cpd]['subgraph'][node_type] = [node_num]

            else:

                self.subgraphs[cpd]['subgraph'][node_type].append(node_num)

    def add2features(self, node_edge, feature, type):

        if type=='node':
            
            node = node_edge.GetSymbol()

            if node not in self.node_features.keys():

                self.node_features[node] = []

                self.node_features[node].append(feature)

            else:

                self.node_features[node].append(feature)

        elif type=='edge':

            edge = self.get_bond_type(node_edge)

            if edge not in self.edge_features.keys():

                self.edge_features[edge] = []

                self.edge_features[edge].append(feature)

            else:

                self.edge_features[edge].append(feature)

        else:

            print("Something error with the input!")

    def get_node_idx(self, node_num, node_type):

        if node_num not in self.node_idx['node_num']:
            
            self.node_idx['node_num'].append(node_num)

            if node_type not in self.node_idx.keys():

                self.node_idx[node_type] = [node_num]

            else:

                self.node_idx[node_type].append(node_num)

        return len(self.node_idx[node_type])-1

    def add2graph_dict(self, src_node, edge, dst_node, nsrc, ndst):

        key = (src_node, edge, dst_node)

        if key not in self.graph_dict.keys():

            self.graph_dict[key] = {
                'u':[],
                'v':[]
            }
            
            # u
            self.graph_dict[key]['u'].append(self.get_node_idx(nsrc, src_node))
            #v
            self.graph_dict[key]['v'].append(self.get_node_idx(ndst, dst_node))

        else:

            # u
            self.graph_dict[key]['u'].append(self.get_node_idx(nsrc, src_node))
            #v
            self.graph_dict[key]['v'].append(self.get_node_idx(ndst, dst_node))


    def get_graph(self):

        for key in self.graph_dict.keys():

            self.graph_data[key] = (torch.tensor(self.graph_dict[key]['u']), 
                                    torch.tensor(self.graph_dict[key]['v']))
            
        G = dgl.heterograph(self.graph_data)

        for key in self.node_features.keys():

            G.nodes[key].data['feature'] = torch.from_numpy(np.array(self.node_features[key]))
        
        # 边只需要知道类型就ok了 不需要知道别的了 
        # for key in self.edge_features.keys():

        #     G.edges[key].data['label'] = self.edge_features[key]

        return G


    def smile2graph(self):

        # self.atom_idx = 0

        for cpd in self.compounds:

            smile = self.compounds[cpd]['smile']

            molecule = Chem.MolFromSmiles(smile)

            n_atom = molecule.GetNumAtoms()

            self.init_subgraph(cpd)

            if n_atom == 1:

                atom = molecule.GetAtomWithIdx(0) 

                # 原子特征
                atom_features = self.get_atom_features(atom) 

                self.add2features(atom, atom_features, type='node')

                # 原子 加入节点

                self.get_node_idx(self.atom_idx, atom.GetSymbol())

                self.add2subgraph(cpd, self.atom_idx, atom.GetSymbol())

                self.atom_idx += 1

            else:

                for i in range(molecule.GetNumAtoms()):

                    atom_i = molecule.GetAtomWithIdx(i) 

                    atom_i_features = self.get_atom_features(atom_i) 

                    self.add2features(atom_i, atom_i_features, type='node')

                    self.add2subgraph(cpd, i + self.atom_idx, atom_i.GetSymbol())

                    for j in range(i+1, molecule.GetNumAtoms()):

                        atom_j = molecule.GetAtomWithIdx(j) 

                        bond_ij = molecule.GetBondBetweenAtoms(i, j)

                        if bond_ij is not None:

                            # bond_ij
                            bond_ij_feature = self.get_bond_features(bond_ij) 
                            self.add2features(bond_ij, bond_ij_feature, type='edge')

                            self.add2graph_dict(atom_i.GetSymbol(), 
                                                self.get_bond_type(bond_ij), 
                                                atom_j.GetSymbol(), 
                                                i + self.atom_idx, 
                                                j + self.atom_idx)
                        
                self.atom_idx += molecule.GetNumAtoms()

        
        return self.get_graph()

        



if __name__ == '__main__':      

    # smile = 'CC(=NCCCCC(C(=O)O)N)N' 
    #'CCC1(CCC(=O)NC1=O)C2=CC=CC=C2' 
    #'C1=CC(=CC=C1NC(C(C(C(C(CO)O)O)O)O)S(=O)(=O)O)S(=O)(=O)C2=CC=C(C=C2)NC(C(C(C(C(CO)O)O)O)O)S(=O)(=O)O'

    cpd_file = '/home/cao/project/dG_GNN/data/KEGG_COMPOUNDS_SMILES/compounds_all.json'
    with open(cpd_file, 'r') as f:
        compounds = json.load(f)

    G = Smile2Graph(compounds)

    print(G.G)

















