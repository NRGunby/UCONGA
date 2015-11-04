import os.path
from csv import reader as csv_reader
from math import radians
# File locations
UCONGA_root = os.path.join('/Users', 'Nate', 'Documents', 'U-CONGA')
pt_file = os.path.join(UCONGA_root, 'periodic_table.csv')
# Parse the periodic table csv file
periodic_table = {}
periodic_list = []  # Get symbol given number, hence the 0th entry

with open(pt_file, 'rb') as pt:
    pt_reader = csv_reader(pt)
    for row in pt_reader:
        periodic_list.append(row[1])
        periodic_table[row[1]] = {'num': int(row[0]), 'vdw': float(row[2])}
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
