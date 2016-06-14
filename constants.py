from math import radians
# van der Waals radii ,atomic numbers, and masses
periodic_table = {'Xx': {'num':  0, 'vdw': 0.0,  'mass':  0.0},     'H': {'num':  1, 'vdw': 1.1,  'mass':  1.0079},
                  'He': {'num':  2, 'vdw': 1.4,  'mass':  4.00260},'Li': {'num':  3, 'vdw': 1.81, 'mass':  6.941},
                  'Be': {'num':  4, 'vdw': 1.53, 'mass':  9.01218}, 'B': {'num':  5, 'vdw': 1.92, 'mass': 10.811},
                   'C': {'num':  6, 'vdw': 1.7,  'mass': 12.011},   'N': {'num':  7, 'vdw': 1.55, 'mass': 14.00674},
                   'O': {'num':  8, 'vdw': 1.52, 'mass': 15.994},   'F': {'num':  9, 'vdw': 1.47, 'mass': 18.998403},
                  'Ne': {'num': 10, 'vdw': 1.54, 'mass': 20.1797}, 'Na': {'num': 11, 'vdw': 2.27, 'mass': 22.989768},
                  'Mg': {'num': 12, 'vdw': 1.73, 'mass': 24.305},  'Al': {'num': 13, 'vdw': 1.84, 'mass': 26.981539},
                  'Si': {'num': 14, 'vdw': 2.1,  'mass': 28.0855},  'P': {'num': 15, 'vdw': 1.8,  'mass': 30.973762},
                   'S': {'num': 16, 'vdw': 1.8,  'mass': 32.066},  'Cl': {'num': 17, 'vdw': 1.75, 'mass': 35.4527},
                  'Ar': {'num': 18, 'vdw': 1.88, 'mass': 39.948},   'K': {'num': 19, 'vdw': 2.75, 'mass': 39.0983},
                  'Ca': {'num': 20, 'vdw': 2.31, 'mass': 40.078},  'Sc': {'num': 21, 'vdw': 2.3,  'mass': 44.95591},
                  'Ti': {'num': 22, 'vdw': 2.15, 'mass': 47.88},    'V': {'num': 23, 'vdw': 2.05, 'mass': 50.9415},
                  'Cr': {'num': 24, 'vdw': 2.05, 'mass': 51.9961}, 'Mn': {'num': 25, 'vdw': 2.05, 'mass': 54.9938},
                  'Fe': {'num': 26, 'vdw': 2.05, 'mass': 55.847},  'Co': {'num': 27, 'vdw': 2.0,  'mass': 58.9332},
                  'Ni': {'num': 28, 'vdw': 2.0,  'mass': 58.6934}, 'Cu': {'num': 29, 'vdw': 2.0,  'mass': 63.546},
                  'Zn': {'num': 30, 'vdw': 2.1,  'mass': 65.39},   'Ga': {'num': 31, 'vdw': 1.87, 'mass': 69.372},
                  'Ge': {'num': 32, 'vdw': 2.11, 'mass': 72.64},   'As': {'num': 33, 'vdw': 1.85, 'mass': 74.92159},
                  'Se': {'num': 34, 'vdw': 1.9,  'mass': 78.96},   'Br': {'num': 35, 'vdw': 1.83, 'mass': 79.904},
                  'Kr': {'num': 36, 'vdw': 2.02, 'mass': 83.80},   'Rb': {'num': 37, 'vdw': 3.03, 'mass': 85.4678},
                  'Sr': {'num': 38, 'vdw': 2.49, 'mass': 87.62},    'Y': {'num': 39, 'vdw': 2.4,  'mass': 88.90585},
                  'Zr': {'num': 40, 'vdw': 2.3,  'mass': 91.224},  'Nb': {'num': 41, 'vdw': 2.15, 'mass': 92.90638},
                  'Mo': {'num': 42, 'vdw': 2.1,  'mass': 95.94},   'Tc': {'num': 43, 'vdw': 2.05, 'mass': 98.9072},
                  'Ru': {'num': 44, 'vdw': 2.05, 'mass':101.07},   'Rh': {'num': 45, 'vdw': 2.0, 'mass':102.9055},
                  'Pd': {'num': 46, 'vdw': 2.05, 'mass':106.42},   'Ag': {'num': 47, 'vdw': 2.1, 'mass':107.8682},
                  'Cd': {'num': 48, 'vdw': 2.2, 'mass':112.411},   'In': {'num': 49, 'vdw': 2.2, 'mass':114.818},
                  'Sn': {'num': 50, 'vdw': 1.93, 'mass':1118.71},  'Sb': {'num': 51, 'vdw': 2.17, 'mass':121.76},
                  'Te': {'num': 52, 'vdw': 2.06, 'mass':127.6},     'I': {'num': 53, 'vdw': 1.98, 'mass':126.90447},
                  'Xe': {'num': 54, 'vdw': 2.16, 'mass':131.29},   'Cs': {'num': 55, 'vdw': 3.43, 'mass':132.90543},
                  'Ba': {'num': 56, 'vdw': 2.68, 'mass':137.327},  'La': {'num': 57, 'vdw': 2.5,  'mass':138.9055},
                  'Ce': {'num': 58, 'vdw': 2.48, 'mass':140.115},  'Pr': {'num': 59, 'vdw': 2.47, 'mass':1140.90765},
                  'Nd': {'num': 60, 'vdw': 2.45, 'mass':144.24},   'Pm': {'num': 61, 'vdw': 2.43, 'mass':144.9127},
                  'Sm': {'num': 62, 'vdw': 2.42, 'mass':150.36},   'Eu': {'num': 63, 'vdw': 2.4,  'mass':151.9655},
                  'Gd': {'num': 64, 'vdw': 2.38, 'mass':157.25},   'Tb': {'num': 65, 'vdw': 2.37, 'mass':158.92534},
                  'Dy': {'num': 66, 'vdw': 2.35, 'mass':162.50},   'Ho': {'num': 67, 'vdw': 2.33, 'mass':164.93032},
                  'Er': {'num': 68, 'vdw': 2.32, 'mass':167.26},   'Tm': {'num': 69, 'vdw': 2.3,  'mass':168.93421},
                  'Yb': {'num': 70, 'vdw': 2.28, 'mass':173.04},   'Lu': {'num': 71, 'vdw': 2.27, 'mass':174.967},
                  'Hf': {'num': 72, 'vdw': 2.25, 'mass':178.49},   'Ta': {'num': 73, 'vdw': 2.2,  'mass':180.9479},
                   'W': {'num': 74, 'vdw': 2.1,  'mass':183.85},   'Re': {'num': 75, 'vdw': 2.05, 'mass':186.207},
                  'Os': {'num': 76, 'vdw': 2.0,  'mass':190.23},   'Ir': {'num': 77, 'vdw': 2.0,  'mass':192.22},
                  'Pt': {'num': 78, 'vdw': 2.05, 'mass':195.08},   'Au': {'num': 79, 'vdw': 2.1,  'mass':196.9665},
                  'Hg': {'num': 80, 'vdw': 2.05, 'mass':200.59},   'Tl': {'num': 81, 'vdw': 1.96, 'mass':204.3833},
                  'Pb': {'num': 82, 'vdw': 2.02, 'mass':207.2},    'Bi': {'num': 83, 'vdw': 2.07, 'mass':208.98037},
                  'Po': {'num': 84, 'vdw': 1.97, 'mass':208.98037},'At': {'num': 85, 'vdw': 2.02, 'mass':209.9871},
                  'Rn': {'num': 86, 'vdw': 2.2,  'mass':222.0176}, 'Fr': {'num': 87, 'vdw': 3.48, 'mass':223.0197},
                  'Ra': {'num': 88, 'vdw': 2.83, 'mass':226.0254}, 'Ac': {'num': 89, 'vdw': 2.0,  'mass':227.0278},
                  'Th': {'num': 90, 'vdw': 2.4,  'mass':232.0381}, 'Pa': {'num': 91, 'vdw': 2.0,  'mass':231.03588},
                   'U': {'num': 92, 'vdw': 2.3,  'mass':238.0289}, 'Np': {'num': 93, 'vdw': 2.0,  'mass':237.0482},
                  'Pu': {'num': 94, 'vdw': 2.0,  'mass':244.0642}, 'Am': {'num': 95, 'vdw': 2.0,  'mass':243.0614},
                  'Cm': {'num': 96, 'vdw': 2.0,  'mass':247.0703}, 'Bk': {'num': 97, 'vdw': 2.0,  'mass':247.0703},
                  'Cf': {'num': 98, 'vdw': 2.0,  'mass':251.0796}, 'Es': {'num': 99, 'vdw': 2.0,  'mass':254},
                  'Fm': {'num':100, 'vdw': 2.0,  'mass':257.0951}, 'Md': {'num':101, 'vdw': 2.0,  'mass':258.1},
                  'No': {'num':102, 'vdw': 2.0,  'mass':259.1009}, 'Lr': {'num':103, 'vdw': 2.0,  'mass':262},
                  'Rf': {'num':104, 'vdw': 2.0,  'mass':261},      'Db': {'num':105, 'vdw': 2.0,  'mass':262},
                  'Sg': {'num':106, 'vdw': 2.0,  'mass':266},      'Bh': {'num':107, 'vdw': 2.0,  'mass':264},
                  'Hs': {'num':108, 'vdw': 2.0,  'mass':269},      'Mt': {'num':109, 'vdw': 2.0,  'mass':268},
                  'Ds': {'num':110, 'vdw': 2.0,  'mass':269},      'Rg': {'num':111, 'vdw': 2.0,  'mass':272},
                  'Cn': {'num':112, 'vdw': 2.0,  'mass':277},     'Uut': {'num':113, 'vdw': 2.0,  'mass':286},
                  'Fl': {'num':114, 'vdw': 2.0,  'mass':289},     'Uup': {'num':115, 'vdw': 2.0,  'mass':289},
                  'Lv': {'num':116, 'vdw': 2.0,  'mass':298}}

periodic_list = ['Xx', 'H', 'He',
                 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
                 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
                 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
                    'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
                 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
                    'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
                 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn',
                 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Uut', 'Fl', 'Uup', 'Lv']
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
