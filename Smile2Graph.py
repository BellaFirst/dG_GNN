import warnings

from rdkit import Chem

# node
atom_vocab = ["H", "B", "C", "N", "O", "F", "Mg", "Si", "P", "S", "Cl", "Cu", "Zn", "Se", "Br", "Sn", "I"]
atom_vocab = {a: i for i, a in enumerate(atom_vocab)}
degree_vocab = range(7)
num_hs_vocab = range(7)
formal_charge_vocab = range(-5, 6)
chiral_tag_vocab = range(4)
total_valence_vocab = range(8)
num_radical_vocab = range(8)
hybridization_vocab = range(len(Chem.rdchem.HybridizationType.values))

atom2valence = {1: 1, 
                5: 3, 
                6: 4, 
                7: 3, 
                8: 2, 
                9: 1, 
                14: 4, 
                15: 5, 
                16: 6, 
                17: 1, 
                35: 1, 
                53: 7}
atom2volume = {'H': 14.4, 'Li': 13.1, 'Na': 23.7, 'K': 45.46, 'Rb': 55.9, 'Cs': 71.07, 'Be': 5.0, 'Mg': 13.9,'Ca': 29.9, 'Sr': 33.7,
                'Ba': 39.24, 'Sc': 15.0, 'Y': 19.8, 'Ti': 10.64, 'Zr': 14.1, 'Hf': 13.6, 'V': 8.78, 'Nb': 10.87,'Ta': 10.9, 'Cr': 7.23,
                'Mo': 9.4, 'W': 9.53, 'Mn': 1.39, 'Tc': 8.5, 'Re': 8.85, 'Fe': 7.1, 'Ru': 8.3, 'Os': 8.49,'Co': 6.7, 'Rh': 8.3, 'Ir': 8.54,
                'Ni': 6.59, 'Pd': 8.9, 'Pt': 9.1, 'Cu': 7.1, 'Ag': 10.3, 'Au': 10.2, 'Zn': 9.2, 'Cd': 13.1,'Hg': 14.82, 'B': 4.6, 'Al': 10.0,
                'Ga': 11.8, 'In': 15.7, 'Tl': 7.2, 'C': 4.58, 'Si': 12.1, 'Ge': 13.6, 'Sn': 16.3, 'Pb': 18.17,'N': 17.3, 'P': 17.0, 'As': 13.1,
                'Sb': 18.23, 'Bi': 21.3, 'O': 14.0, 'S': 15.5, 'Se': 16.45, 'Te': 20.5, 'Po': 22.23, 'F': 17.1,'Cl': 22.7, 'Br': 23.5,
                'I': 25.74, 'La': 20.73, 'Ce': 20.67, 'Pr': 20.8, 'Nd': 20.6, 'Pm': 28.9, 'Sm': 19.95, 'Eu': 28.9,'Gd': 19.9, 'Tb': 19.2,
                'Dy': 19.0, 'Ho': 18.7, 'Er': 18.4, 'Tm': 18.1, 'Yb': 24.79, 'Lu': 17.78, 'Ne': 16.7, 'Ar': 28.5,'Kr': 38.9, 'Xe': 37.3,
                'Rn': 50.5, 'Ra': 45.2, 'Ac': 22.54, 'Th': 19.9, 'Pa': 15, 'U': 12.59, 'Np': 11.62, 'Pu': 12.32,'Am': 17.86, 'Cm': 18.28, 
                '*': 0},

def atom_position(atom):
    """
    Atom position.
    Return 3D position if available, otherwise 2D position is returned.
    """
    mol = atom.GetOwningMol()
    if mol.GetNumConformers() == 0:
        mol.Compute2DCoords()
    conformer = mol.GetConformer()
    pos = conformer.GetAtomPosition(atom.GetIdx())
    return [pos.x, pos.y, pos.z]

# edge
bond2id = {"ELSE": 0, "SINGLE": 1, "DOUBLE": 2, "TRIPLE": 3, "AROMATIC": 4}
id2bond = {v: k for k, v in bond2id.items()}
bond_type_vocab = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
                    Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
bond_type_vocab = {b: i for i, b in enumerate(bond_type_vocab)}
bond_dir_vocab = range(len(Chem.rdchem.BondDir.values))
bond_stereo_vocab = range(len(Chem.rdchem.BondStereo.values))


def onehot(x, vocab, allow_unknown=False):
    if x in vocab:
        if isinstance(vocab, dict):
            index = vocab[x]
        else:
            index = vocab.index(x)
    else:
        index = -1
    if allow_unknown:
        feature = [0] * (len(vocab) + 1)
        if index == -1:
            warnings.warn("Unknown value `%s`" % x)
        feature[index] = 1
    else:
        feature = [0] * len(vocab)
        if index == -1:
            raise ValueError("Unknown value `%s`. Available vocabulary is `%s`" % (x, vocab))
        feature[index] = 1

    return feature

class Smile2Graph(object):

    def __init__(self, smile) -> None:

        self.smile = smile
        self.mol = Chem.MolFromSmiles(self.smile)
        if self.mol is None:
            raise ValueError("Invalid SMILES `%s`" % self.smile)
        else:
            self.mol = Chem.MolFromSmiles("")


    def _atom_feature(self):
        """Default atom feature.

            Features:
                GetSymbol(): one-hot embedding for the atomic symbol 原子符号
                
                GetChiralTag(): one-hot embedding for atomic chiral tag 
                
                GetTotalDegree(): one-hot embedding for the degree of the atom in the molecule including Hs
                
                GetFormalCharge(): one-hot embedding for the number of formal charges in the molecule
                
                GetTotalNumHs(): one-hot embedding for the total number of Hs (explicit and implicit) on the atom H原子
                
                GetNumRadicalElectrons(): one-hot embedding for the number of radical electrons on the atom
                
                GetHybridization(): one-hot embedding for the atom's hybridization
                
                GetIsAromatic(): whether the atom is aromatic
                
                IsInRing(): whether the atom is in a ring
                
                atom_position(): the 3D position of the atom
        """
        pass

    def _bond_feature(self):
        pass

    
    def get_features(self):

        # Node <--- atom
        atoms = [self.mol.GetAtomWithIdx(i) for i in range(self.mol.GetNumAtoms())]
        node_feature = []
        for atom in atoms:
            node_feature.append(onehot(atom.GetSymbol(), atom_vocab, allow_unknown=True) + \
            onehot(atom.GetChiralTag(), chiral_tag_vocab) + \
            onehot(atom.GetTotalDegree(), degree_vocab, allow_unknown=True) + \
            onehot(atom.GetFormalCharge(), formal_charge_vocab) + \
            onehot(atom.GetTotalNumHs(), num_hs_vocab) + \
            onehot(atom.GetNumRadicalElectrons(), num_radical_vocab) + \
            onehot(atom.GetHybridization(), hybridization_vocab) + \
            [atom.GetIsAromatic(), atom.IsInRing()] + \
            atom_position(atom))
