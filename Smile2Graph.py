import json
from rdkit import Chem
from rdkit.Chem import AllChem
from matplotlib import pyplot as plt
from torchdrug.data.rdkit import draw


cpd_file = '/home/caoyh/project/dG_GNN/data/KEGG_COMPOUNDS_SMILES/compounds_all.json'
with open(cpd_file, 'r') as f:
    compounds = json.load(f)

smile_list = [compounds[cpd]['smile'] for cpd in compounds]


for cpd in compounds:

    smile = compounds[cpd]['smile']

    molecule = Chem.MolFromSmiles(smile)

# atom
atom_vocab = ["H", "B", "C", "N", "O", "F", "Mg", "Si", "P", "S", "Cl", "Cu", "Zn", "Se", "Br", "Sn", "I"]
atom_vocab = {a: i for i, a in enumerate(atom_vocab)}
atomic_radius = {'H': 0.79, 'Li': 2.05, 'Na': 2.23, 'K': 2.77, 'Rb': 2.98, 'Cs': 3.34, 'Be': 1.4, 'Mg': 1.72,
                     'Ca': 2.23, 'Sr': 2.45, 'Ba': 2.78, 'Sc': 2.09,
                     'Y': 2.27, 'Ti': 2, 'Zr': 2.16, 'Hf': 2.16, 'V': 1.92, 'Nb': 2.08, 'Ta': 2.09, 'Cr': 1.85,
                     'Mo': 2.01, 'W': 2.02, 'Mn': 1.79, 'Tc': 1.95,
                     'Re': 1.97, 'Fe': 1.72, 'Ru': 1.89, 'Os': 1.92, 'Co': 1.67, 'Rh': 1.83, 'Ir': 1.87, 'Ni': 1.62,
                     'Pd': 1.79, 'Pt': 1.83, 'Cu': 1.57, 'Ag': 1.75,
                     'Au': 1.79, 'Zn': 1.53, 'Cd': 1.71, 'Hg': 1.76, 'B': 1.17, 'Al': 1.82, 'Ga': 1.81, 'In': 2,
                     'Tl': 2.08, 'C': 0.91, 'Si': 1.46, 'Ge': 1.52, 'Sn': 1.72,
                     'Pb': 1.81, 'N': 0.75, 'P': 1.23, 'As': 1.33, 'Sb': 1.53, 'Bi': 1.63, 'O': 0.65, 'S': 1.09,
                     'Se': 1.22, 'Te': 1.42, 'Po': 1.53, 'F': 0.57, 'Cl': 0.97, 'Br': 1.12,
                     'I': 1.32, 'At': 1.43, 'La': 2.74, 'Ce': 2.7, 'Pr': 2.67, 'Nd': 2.64, 'Pm': 2.62, 'Eu': 2.56,
                     'Gd': 2.54, 'Tb': 2.51, 'Dy': 2.49, 'Ho': 2.47, 'Er': 2.45,
                     'Tm': 2.42, 'Yb': 2.4, 'Lu': 2.25, 'He': 0.49, 'Ne': 0.51, 'Ar': 0.88, 'Kr': 1.03, 'Xe': 1.24,
                     'Rn': 1.34, 'Fr': 1.8, 'Ra': 1.43, 'Ac': 1.119, 'Th': 0.972, 'Pa': 0.78, 'U': 0.52, 'Np': 0.75,
                     'Pu': 0.887,
                     'Am': 0.982, 'Cm': 0.97, 'Bk': 0.949, 'Cf': 0.934, 'Es': 0.925}
atomic_volume = {'H': 14.4, 'Li': 13.1, 'Na': 23.7, 'K': 45.46, 'Rb': 55.9, 'Cs': 71.07, 'Be': 5.0, 'Mg': 13.9,
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
                     'Am': 17.86, 'Cm': 18.28}                     
chiral_tag_vocab = {Chem.rdchem.ChiralType.values[i]: i for i in range(len(Chem.rdchem.ChiralType.values))}
hybridization_vocab = {Chem.rdchem.HybridizationType.values[i]: i for i in range(len(Chem.rdchem.HybridizationType.values))}
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

# bond
bond_type_vocab = [Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
                   Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
bond_type_vocab = {b: i for i, b in enumerate(bond_type_vocab)}
bond_dir_vocab = {Chem.rdchem.BondDir.values[i]: i for i in range(len(Chem.rdchem.BondDir.values))}
bond_stereo_vocab = {Chem.rdchem.BondStereo.values[i]: i for i in range(len(Chem.rdchem.BondStereo.values))}

def bond_length(bond):
    """Bond length"""
    mol = bond.GetOwningMol()
    if mol.GetNumConformers() == 0:
        mol.Compute2DCoords()
    conformer = mol.GetConformer()
    h = conformer.GetAtomPosition(bond.GetBeginAtomIdx())
    t = conformer.GetAtomPosition(bond.GetEndAtomIdx())
    return [h.Distance(t)]

atom2valence = {1: 1, 5: 3, 6: 4, 7: 3, 8: 2, 9: 1, 14: 4, 15: 5, 16: 6, 17: 1, 35: 1, 53: 7}
bond2valence = [1, 2, 3, 1.5]
bond2id = {"SINGLE": 0, "DOUBLE": 1, "TRIPLE": 2, "AROMATIC": 3}
id2bond = {v: k for k, v in bond2id.items()}


class Smile2Graph(object):

    def __init__(self, smile, with_hydrogen=False, kekulize=False) -> None:
        self.smile = smile
        self.with_hydrogen = with_hydrogen
        self.kekulze = kekulize

        self.mol = self._init_mol()

        self.empty_mol = Chem.MolFromSmiles("")
        self.dummy_mol = Chem.MolFromSmiles("CC")
        self.dummy_atom = self.dummy_mol.GetAtomWithIdx(0)
        self.dummy_bond = self.dummy_mol.GetBondWithIdx(0)

        self.atoms = [self.mol.GetAtomWithIdx(i) for i in range(self.mol.GetNumAtoms())] + [self.dummy_atom]
        self.bonds = [self.mol.GetBondWithIdx(i) for i in range(self.mol.GetNumBonds())] + [self.dummy_bond]


    def _init_mol(self):

        mol = Chem.MolFromSmiles(self.smile)

        if mol is None:
            raise ValueError("Invalid SMILES `%s`" % self.smile)

        if self.with_hydrogen:
            mol = Chem.AddHs(mol)

        if self.kekulze:
            Chem.Kekulize(mol)

        return mol

    def get_atom_features(self, atom):

        return [
            atom.GetAtomicNum(),
            atom_vocab(atom.GetSymbol()),
            atomic_radius(atom.GetSymbol()),
            atomic_volume(atom.GetSymbol()),
            chiral_tag_vocab[atom.GetChiralTag()],
            atom.GetTotalValence(),
            atom.GetFormalCharge(),
            atom.GetDegree(), 
            atom.GetTotalDegree(),
            atom.GetTotalNumHs(),
            atom.GetNumExplicitHs(),
            atom.GetNumRadicalElectrons(),
            hybridization_vocab[atom.GetHybridization()],
            atom.GetAtomMapNum(),
            int(atom.GetIsAromatic()),
            int(atom.IsInRing()),  
        ] + atom_position(atom)

        
    
    def get_edge_features(self, bond):

        return [
            bond.GetBeginAtomIdx(), 
            bond_type_vocab[bond.GetBondType()],
            bond.GetEndAtomIdx(),
            bond_dir_vocab[bond.GetBondDir()],
            bond_stereo_vocab[bond.GetStereo()],
            int(bond.GetIsConjugated()),     
        ] + bond_length(bond)

    def visualize(self, title=None, save_file=None, figure_size=(3, 3), ax=None, atom_map=False):
        """
        Visualize this molecule with matplotlib.

        Parameters:
            title (str, optional): title for this molecule
            save_file (str, optional): ``png`` or ``pdf`` file to save visualization.
                If not provided, show the figure in window.
            figure_size (tuple of int, optional): width and height of the figure
            ax (matplotlib.axes.Axes, optional): axis to plot the figure
            atom_map (bool, optional): visualize atom mapping or not
        """
        is_root = ax is None
        if ax is None:
            fig = plt.figure(figsize=figure_size)
            if title is not None:
                ax = plt.gca()
            else:
                ax = fig.add_axes([0, 0, 1, 1])
        if title is not None:
            ax.set_title(title)

        mol = self.to_molecule()
        if not atom_map:
            for atom in mol.GetAtoms():
                atom.SetAtomMapNum(0)
        draw.MolToMPL(mol, ax=ax)
        ax.set_frame_on(False)

        if is_root:
            if save_file:
                fig.savefig(save_file)
            else:
                fig.show()



