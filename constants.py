from math import radians
# van der Waals radii and
periodic_table = { 'Xx': {'num': 0, 'vdw': 0.0}, 'H': {'num': 1, 'vdw': 1.1},
                'He': {'num': 2, 'vdw': 1.4}, 'Li': {'num': 3, 'vdw': 1.81}, 'Be': {'num': 4, 'vdw': 1.53},
                'B': {'num': 5, 'vdw': 1.92}, 'C': {'num': 6, 'vdw': 1.7}, 'N': {'num': 7, 'vdw': 1.55},
                'O': {'num': 8, 'vdw': 1.52}, 'F': {'num': 9, 'vdw': 1.47}, 'Ne': {'num': 10, 'vdw': 1.54},
                'Na': {'num': 11, 'vdw': 2.27}, 'Mg': {'num': 12, 'vdw': 1.73}, 'Al': {'num': 13, 'vdw': 1.84},
                'Si': {'num': 14, 'vdw': 2.1}, 'P': {'num': 15, 'vdw': 1.8}, 'S': {'num': 16, 'vdw': 1.8},
                'Cl': {'num': 17, 'vdw': 1.75}, 'Ar': {'num': 18, 'vdw': 1.88}, 'K': {'num': 19, 'vdw': 2.75},
                'Ca': {'num': 20, 'vdw': 2.31}, 'Sc': {'num': 21, 'vdw': 2.3}, 'Ti': {'num': 22, 'vdw': 2.15},
                'V': {'num': 23, 'vdw': 2.05}, 'Cr': {'num': 24, 'vdw': 2.05},'Mn': {'num': 25, 'vdw': 2.05},
                'Fe': {'num': 26, 'vdw': 2.05}, 'Co': {'num': 27, 'vdw': 2.0}, 'Ni': {'num': 28, 'vdw': 2.0},
                'Cu': {'num': 29, 'vdw': 2.0}, 'Zn': {'num': 30, 'vdw': 2.1}, 'Ga': {'num': 31, 'vdw': 1.87},
                'Ge': {'num': 32, 'vdw': 2.11}, 'As': {'num': 33, 'vdw': 1.85}, 'Se': {'num': 34, 'vdw': 1.9},
                'Br': {'num': 35, 'vdw': 1.83}, 'Kr': {'num': 36, 'vdw': 2.02}, 'Rb': {'num': 37, 'vdw': 3.03},
                'Sr': {'num': 38, 'vdw': 2.49}, 'Y': {'num': 39, 'vdw': 2.4}, 'Zr': {'num': 40, 'vdw': 2.3},
                'Nb': {'num': 41, 'vdw': 2.15}, 'Mo': {'num': 42, 'vdw': 2.1},'Tc': {'num': 43, 'vdw': 2.05},
                'Ru': {'num': 44, 'vdw': 2.05}, 'Rh': {'num': 45, 'vdw': 2.0}, 'Pd': {'num': 46, 'vdw': 2.05},
                'Ag': {'num': 47, 'vdw': 2.1}, 'Cd': {'num': 48, 'vdw': 2.2}, 'In': {'num': 49, 'vdw': 2.2},
                'Sn': {'num': 50, 'vdw': 1.93}, 'Sb': {'num': 51, 'vdw': 2.17}, 'Te': {'num': 52, 'vdw': 2.06},
                'I': {'num': 53, 'vdw': 1.98}, 'Xe': {'num': 54, 'vdw': 2.16}, 'Cs': {'num': 55, 'vdw': 3.43},
                'Ba': {'num': 56, 'vdw': 2.68}, 'La': {'num': 57, 'vdw': 2.5}, 'Ce': {'num': 58, 'vdw': 2.48},
                'Pr': {'num': 59, 'vdw': 2.47}, 'Nd': {'num': 60, 'vdw': 2.45}, 'Pm': {'num': 61, 'vdw': 2.43},
                'Sm': {'num': 62, 'vdw': 2.42}, 'Eu': {'num': 63, 'vdw': 2.4}, 'Gd': {'num': 64, 'vdw': 2.38},
                'Tb': {'num': 65, 'vdw': 2.37}, 'Dy': {'num': 66, 'vdw': 2.35}, 'Ho': {'num': 67, 'vdw': 2.33},
                'Er': {'num': 68, 'vdw': 2.32}, 'Tm': {'num': 69, 'vdw': 2.3}, 'Yb': {'num': 70, 'vdw': 2.28},
                'Lu': {'num': 71, 'vdw': 2.27}, 'Hf': {'num': 72, 'vdw': 2.25}, 'Ta': {'num': 73, 'vdw': 2.2},
                'W': {'num': 74, 'vdw': 2.1}, 'Re': {'num': 75, 'vdw': 2.05}, 'Os': {'num': 76, 'vdw': 2.0},
                'Ir': {'num': 77, 'vdw': 2.0}, 'Pt': {'num': 78, 'vdw': 2.05}, 'Au': {'num': 79, 'vdw': 2.1},
                'Hg': {'num': 80, 'vdw': 2.05}, 'Tl': {'num': 81, 'vdw': 1.96}, 'Pb': {'num': 82, 'vdw': 2.02},
                'Bi': {'num': 83, 'vdw': 2.07}, 'Po': {'num': 84, 'vdw': 1.97},'At': {'num': 85, 'vdw': 2.02},
                'Rn': {'num': 86, 'vdw': 2.2}, 'Fr':{'num': 87, 'vdw': 3.48}, 'Ra': {'num': 88, 'vdw': 2.83},
                'Ac': {'num': 89, 'vdw': 2.0}, 'Th': {'num': 90, 'vdw': 2.4}, 'Pa': {'num': 91, 'vdw': 2.0},
                'U': {'num': 92, 'vdw': 2.3}, 'Np': {'num': 93, 'vdw': 2.0},  'Pu': {'num': 94, 'vdw': 2.0},
                'Am': {'num': 95, 'vdw': 2.0}, 'Cm': {'num': 96, 'vdw': 2.0}, 'Bk': {'num': 97, 'vdw': 2.0},
                'Cf': {'num': 98, 'vdw': 2.0}, 'Es': {'num': 99, 'vdw': 2.0}, 'Fm': {'num': 100, 'vdw': 2.0},
                'Md': {'num': 101, 'vdw': 2.0}, 'No': {'num': 102, 'vdw': 2.0}, 'Lr': {'num': 103, 'vdw': 2.0},
                'Rf': {'num': 104, 'vdw': 2.0}, 'Db': {'num': 105, 'vdw': 2.0}, 'Sg': {'num': 106, 'vdw': 2.0},
                'Bh': {'num': 107, 'vdw': 2.0}, 'Hs': {'num': 108, 'vdw': 2.0}, 'Mt': {'num': 109, 'vdw': 2.0},
                'Ds': {'num': 110, 'vdw': 2.0}, 'Rg': {'num': 111, 'vdw': 2.0}, 'Cn': {'num': 112, 'vdw': 2.0},
                'Uut': {'num': 113, 'vdw': 2.0}, 'Uuq': {'num': 114, 'vdw': 2.0}, 'Uup': {'num': 115, 'vdw': 2.0},
                'Uuh': {'num': 116, 'vdw': 2.0}}

periodic_list = ['Xx', 'H', 'He',
                 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
                 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
                 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
                    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
                 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
                    'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
                 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn',
                 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Uut', 'Uuq', 'Uup', 'Uuh']
# cml file attribute names
lbl_atom = 'atom'
lbl_id = 'id'
lbl_element_type = 'elementType'
lbl_atom_array = 'atomArray'
lbl_bond_array = 'bondArray'
lbl_molecule = 'molecule'
lbl_order = 'order'
lbl_atom_refs = 'atomRefs2'
lbl_bond = 'bond'
lbl_x = 'x3'
lbl_y = 'y3'
lbl_z = 'z3'


def id_to_py(idx):
    '''
        Converts from the indexing used in cml files (1-based)
        to the indexing used internally (0-based)
        '''
    return int(idx[1:]) - 1


def py_to_id(idx):
    '''
        Converts from the indexing used internally (0-based)
        to the indexing used in cml files (1-based)
        '''
    return 'a' + str(1 + idx)
