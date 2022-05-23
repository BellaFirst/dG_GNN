import math
import warnings

from matplotlib import pyplot as plt
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
import torch
from torch_scatter import scatter_add, scatter_min, scatter_max

from torchdrug import utils
from torchdrug.data import constant, Graph, PackedGraph
from torchdrug.core import Registry as R
from torchdrug.data.rdkit import draw
from torchdrug.utils import pretty

plt.switch_backend("agg")

class MyMolecule(Graph):
    """
    Molecule graph with chemical features.

    Parameters:
        edge_list (array_like, optional): list of edges of shape :math:`(|E|, 3)`.
            Each tuple is (node_in, node_out, bond_type).
        atom_type (array_like, optional): atom types of shape :math:`(|V|,)`
        bond_type (array_like, optional): bond types of shape :math:`(|E|,)`
        formal_charge (array_like, optional): formal charges of shape :math:`(|V|,)`
        explicit_hs (array_like, optional): number of explicit hydrogens of shape :math:`(|V|,)`
        chiral_tag (array_like, optional): chirality tags of shape :math:`(|V|,)`
        radical_electrons (array_like, optional): number of radical electrons of shape :math:`(|V|,)`
        atom_map (array_likeb optional): atom mappings of shape :math:`(|V|,)`
        bond_stereo (array_like, optional): bond stereochem of shape :math:`(|E|,)`
        stereo_atoms (array_like, optional): ids of stereo atoms of shape :math:`(|E|,)`
    """

    bond2id = {"SINGLE": 0, "DOUBLE": 1, "TRIPLE": 2, "AROMATIC": 3}
    atom2valence = {1: 1, 5: 3, 6: 4, 7: 3, 8: 2, 9: 1, 14: 4, 15: 5, 16: 6, 17: 1, 35: 1, 53: 7}
    bond2valence = [1, 2, 3, 1.5]
    id2bond = {v: k for k, v in bond2id.items()}
    empty_mol = Chem.MolFromSmiles("")
    dummy_mol = Chem.MolFromSmiles("CC")
    dummy_atom = dummy_mol.GetAtomWithIdx(0)
    dummy_bond = dummy_mol.GetBondWithIdx(0)

    def __init__(self, 
                 edge_list=None, 
                 atom_type=None, 
                 bond_type=None, 
                 formal_charge=None, 
                 explicit_hs=None,
                 chiral_tag=None, 
                 radical_electrons=None, 
                 atom_map=None, 
                 bond_stereo=None, 
                 stereo_atoms=None,
                 node_position=None, 
                 **kwargs):

        if "num_relation" not in kwargs:
            kwargs["num_relation"] = len(self.bond2id)
        super(MyMolecule, self).__init__(edge_list=edge_list, **kwargs)

        atom_type, bond_type = self._standarize_atom_bond(atom_type, bond_type)

        formal_charge = self._standarize_attribute(formal_charge, self.num_node)
        explicit_hs = self._standarize_attribute(explicit_hs, self.num_node)
        chiral_tag = self._standarize_attribute(chiral_tag, self.num_node)
        radical_electrons = self._standarize_attribute(radical_electrons, self.num_node)
        atom_map = self._standarize_attribute(atom_map, self.num_node)
        bond_stereo = self._standarize_attribute(bond_stereo, self.num_edge)
        stereo_atoms = self._standarize_attribute(stereo_atoms, (self.num_edge, 2))

        if node_position is not None:
            node_position = torch.as_tensor(node_position, dtype=torch.float, device=self.device)

        with self.atom():
            self.atom_type = atom_type
            self.formal_charge = formal_charge
            self.explicit_hs = explicit_hs
            self.chiral_tag = chiral_tag
            self.radical_electrons = radical_electrons
            self.atom_map = atom_map
            if node_position is not None:
                self.node_position = node_position

        with self.bond():
            self.bond_type = bond_type
            self.bond_stereo = bond_stereo
            self.stereo_atoms = stereo_atoms

    def _standarize_atom_bond(self, atom_type, bond_type):
        if atom_type is None:
            raise ValueError("`atom_type` should be provided")
        if bond_type is None:
            raise ValueError("`bond_type` should be provided")

        atom_type = torch.as_tensor(atom_type, dtype=torch.long, device=self.device)
        bond_type = torch.as_tensor(bond_type, dtype=torch.long, device=self.device)
        return atom_type, bond_type

    def _standarize_attribute(self, attribute, size):
        if attribute is not None:
            attribute = torch.as_tensor(attribute, dtype=torch.long, device=self.device)
        else:
            if isinstance(size, torch.Tensor):
                size = size.tolist()
            attribute = torch.zeros(size, dtype=torch.long, device=self.device)
        return attribute

    @classmethod
    def _standarize_option(cls, option):
        if option is None:
            option = []
        elif isinstance(option, str):
            option = [option]
        return option

    def _check_no_stereo(self):
        if (self.bond_stereo > 0).any():
            warnings.warn("Try to apply masks on molecules with stereo bonds. This may produce invalid molecules. "
                          "To discard stereo information, call `mol.bond_stereo[:] = 0` before applying masks.")

    def _maybe_num_node(self, edge_list):
        if len(edge_list):
            return edge_list[:, :2].max().item() + 1
        else:
            return 0

    @classmethod # 类方法（不需要实例化类就可以被类本身调用）
    def from_molecule(cls, mol, node_feature="default", edge_feature="default", graph_feature=None,
                      with_hydrogen=False, kekulize=False):
        """
        Create a molecule from a RDKit object.

        Parameters:
            mol (rdchem.Mol): molecule
            node_feature (str or list of str, optional): node features to extract
            edge_feature (str or list of str, optional): edge features to extract
            graph_feature (str or list of str, optional): graph features to extract
            with_hydrogen (bool, optional): store hydrogens in the molecule graph. 在分子图中存储氢
                By default, hydrogens are dropped 氢被丢弃
            kekulize (bool, optional): convert aromatic bonds to single/double bonds.
                Note this only affects the relation in ``edge_list``.
                For ``bond_type``, aromatic bonds are always stored explicitly.
                By default, aromatic bonds are stored.
            kekulize (bool, optional): 将芳香键转换为单/双键。
                 请注意，这只影响 ``edge_list`` 中的关系。
                 对于 ``bond_type``，芳香键总是被显式存储。
                 默认情况下，芳烃键被存储。
        """
        
        if mol is None:
            mol = cls.empty_mol

        if with_hydrogen: # 在分子图中存储氢
            mol = Chem.AddHs(mol)
        if kekulize: # 将芳香键转换为单/双键。
            Chem.Kekulize(mol)

        node_feature = cls._standarize_option(node_feature)
        edge_feature = cls._standarize_option(edge_feature)
        graph_feature = cls._standarize_option(graph_feature)

        # Node <--- atom
        atom_type = []
        formal_charge = []
        explicit_hs = []
        chiral_tag = []
        radical_electrons = []
        atom_map = []
        node_position = []
        
        _node_feature = []

        atoms = [mol.GetAtomWithIdx(i) for i in range(mol.GetNumAtoms())] # + [cls.dummy_atom] # 为什么要加cls.dummy_atom
        for atom in atoms:
            atom_type.append(atom.GetAtomicNum()) # 获取原子序号
            formal_charge.append(atom.GetFormalCharge()) # 获取原子形式电荷
            explicit_hs.append(atom.GetNumExplicitHs()) # H原子数量
            chiral_tag.append(atom.GetChiralTag())
            radical_electrons.append(atom.GetNumRadicalElectrons())#自由基电子数量
            atom_map.append(atom.GetAtomMapNum()) # 获取原子的原子地图编号，如果不存在原子图，则返回0。
            feature = []
            for name in node_feature:
                func = R.get("features.atom.%s" % name)
                feature += func(atom)
            _node_feature.append(feature)

        atom_type = torch.tensor(atom_type)[:-1]
        formal_charge = torch.tensor(formal_charge)[:-1]
        explicit_hs = torch.tensor(explicit_hs)[:-1]
        chiral_tag = torch.tensor(chiral_tag)[:-1]
        radical_electrons = torch.tensor(radical_electrons)[:-1]
        atom_map = torch.tensor(atom_map)[:-1]
        if mol.GetNumConformers() > 0:
            node_position = torch.tensor(mol.GetConformer().GetPositions())
        else:
            node_position = None
        if len(node_feature) > 0:
            _node_feature = torch.tensor(_node_feature)[:-1]
        else:
            _node_feature = None

        # Edge <--- bond
        edge_list = []
        bond_type = []
        bond_stereo = []
        stereo_atoms = []
        _edge_feature = []
        bonds = [mol.GetBondWithIdx(i) for i in range(mol.GetNumBonds())] + [cls.dummy_bond]
        for bond in bonds:
            type = str(bond.GetBondType())
            stereo = bond.GetStereo()
            if stereo:
                _atoms = [a for a in bond.GetStereoAtoms()]
            else:
                _atoms = [0, 0]
            if type not in cls.bond2id:
                continue
            type = cls.bond2id[type]
            h, t = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_list += [[h, t, type], [t, h, type]]
            # always explicitly store aromatic bonds, no matter kekulize or not 
            # 始终明确存储芳香键，无论是否 kekulize
            if bond.GetIsAromatic():
                type = cls.bond2id["AROMATIC"]
            bond_type += [type, type]
            bond_stereo += [stereo, stereo]
            stereo_atoms += [_atoms, _atoms]
            feature = []
            for name in edge_feature:
                func = R.get("features.bond.%s" % name)
                feature += func(bond)
            _edge_feature += [feature, feature]
        edge_list = edge_list[:-2]
        bond_type = torch.tensor(bond_type)[:-2]
        bond_stereo = torch.tensor(bond_stereo)[:-2]
        stereo_atoms = torch.tensor(stereo_atoms)[:-2]
        if len(edge_feature) > 0:
            _edge_feature = torch.tensor(_edge_feature)[:-2]
        else:
            _edge_feature = None

        # Graph
        _graph_feature = []
        for name in graph_feature:
            func = R.get("features.molecule.%s" % name)
            _graph_feature += func(mol)
        if len(graph_feature) > 0:
            _graph_feature = torch.tensor(_graph_feature)
        else:
            _graph_feature = None

        num_relation = len(cls.bond2id) - 1 if kekulize else len(cls.bond2id)

        return cls(edge_list, 
                   atom_type, 
                   bond_type,
                   formal_charge=formal_charge, 
                   explicit_hs=explicit_hs,
                   chiral_tag=chiral_tag, 
                   radical_electrons=radical_electrons, 
                   atom_map=atom_map,
                   bond_stereo=bond_stereo, 
                   stereo_atoms=stereo_atoms, 
                   node_position=node_position,
                   node_feature=_node_feature, 
                   edge_feature=_edge_feature, 
                   graph_feature=_graph_feature,
                   num_node=mol.GetNumAtoms(), 
                   num_relation=num_relation)


    @classmethod
    def from_smiles(cls, smiles, node_feature="default", edge_feature="default", graph_feature=None,
                    with_hydrogen=False, kekulize=False):
        """
        Create a molecule from a SMILES string.

        Parameters:
            smiles (str): SMILES string
            node_feature (str or list of str, optional): node features to extract
            edge_feature (str or list of str, optional): edge features to extract
            graph_feature (str or list of str, optional): graph features to extract
            with_hydrogen (bool, optional): store hydrogens in the molecule graph.
                By default, hydrogens are dropped
            kekulize (bool, optional): convert aromatic bonds to single/double bonds.
                Note this only affects the relation in ``edge_list``.
                For ``bond_type``, aromatic bonds are always stored explicitly.
                By default, aromatic bonds are stored.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Invalid SMILES `%s`" % smiles)

        return cls.from_molecule(mol, node_feature, edge_feature, graph_feature, with_hydrogen, kekulize)

    def node_mask(self, index, compact=False):
        self._check_no_stereo()
        return super(MyMolecule, self).node_mask(index, compact)


    def edge_mask(self, index):
        self._check_no_stereo()
        return super(MyMolecule, self).edge_mask(index)


    def undirected(self, add_inverse=False):
        if add_inverse:
            raise ValueError("Bonds are undirected relations, but `add_inverse` is specified")
        return super(MyMolecule, self).undirected(add_inverse)


    def atom(self):
        """
        Context manager for atom attributes.
        """
        return self.node()


    def bond(self):
        """
        Context manager for bond attributes.
        """
        return self.edge()


if __name__=='__main__':

    mol = MyMolecule.from_smiles("C1=CC=CC=C1")
    mol.visualize()
    print(mol.node_feature.shape)
    print(mol.edge_feature.shape)

