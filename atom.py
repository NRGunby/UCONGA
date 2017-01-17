import numpy
from constants import *
import xml.etree.ElementTree as ET


class atom(object):
    '''
    Represents an atom in a molecule
    Design inspired by open babel's OBAtom
    '''
    def __init__(self, num, x, y, z, mol=None):
        self.num = num
        self.coords = numpy.array([x, y, z])
        self.mol = mol
        self.ids_cache = []
        self.get_id_cache = -1

    def get_id(self):
        '''
            Returns the id of a molecule in its molecule's *atoms*.
            This is zero-based
            '''
        if self.mol:
            if self.get_id_cache >= 0:
                return self.get_id_cache
            else:
                self.get_id_cache = self.mol.atoms.index(self)
                return self.get_id_cache
        else:
            raise(ValueError, "Atom not associated with a molecule")

    def get_bond_ids(self):
        '''
        Returns the ids of all atoms bonded to this atom
        '''
        if not self.ids_cache:
            idx_self = self.get_id()
            for each_idx_atom in range(len(self.mol.atoms)):
                if self.mol.get_bond_order(idx_self, each_idx_atom):
                    self.ids_cache.append(each_idx_atom)
        return self.ids_cache[:]

    def get_heavy_valence(self):
        '''
        Returns the number of non-hydrogen atoms bonded to this atom
        '''
        all_ids_bonded = self.get_bond_ids()
        heavy_valence = 0
        for each in all_ids_bonded:
            if self.mol.atoms[each].num > 1:
                heavy_valence += 1
        return heavy_valence

    def search_away_from(self, other_id):
        '''
        Returns the ids of all neighbours except one
        '''
        ids_bonded = self.get_bond_ids()
        ids_bonded.remove(other_id)
        return ids_bonded

    def all_neighbours_away_from(self, *other_ids):
        '''
        Returns all children of an atom
        assuming the atoms with other_ids are its parents
        That is, return all atoms closer to this then to the others
        '''
        distance_matrix = self.mol.distances
        idx_self = self.get_id()
        all_away_neighbours = []
        if len(other_ids) == 0:
            return [i for i in range(len(self.mol.atoms))
                    if self.mol.distances[i][idx_self]]
        for each_idx_atom in range(len(self.mol.atoms)):
            distance_to_others = min([distance_matrix[each_idx_atom][each_idx_other]
                                      for each_idx_other in other_ids])
            if (distance_matrix[each_idx_atom][idx_self] <= distance_to_others
                    and self.mol.distances[each_idx_atom][idx_self]):
                all_away_neighbours.append(each_idx_atom)
        return all_away_neighbours

    def get_vdw(self):
        '''
        Returns the van der Waals radius
        This is probably breaking strict OO design, but OO strictness leads
        to seriously crazy code (see openbabel)
        '''
        return periodic_table[periodic_list[self.num]]['vdw']

    def get_vector(self, at2):
        '''
        Returns the vector that will translate this to another atom
        '''
        return self.coords - at2.coords

    def get_distance(self, at2):
        '''
        Returns the distance to another atom
        '''
        return numpy.linalg.norm(self.get_vector(at2))

    def copy(self):
        '''
        Returns a copy of this atom **not** attached to any molecule
        '''
        return atom(self.num, *self.coords)

    def translate(self, vector):
        '''
        Translate the atom by the supplied 3-vector
        '''
        self.coords = self.coords + vector

    def to_cml(self):
        '''
        Creates a cml atom element
        '''
        element = ET.Element(lbl_atom)
        labels = []
        values = []
        try:
            labels.append(lbl_id)
            values.append(py_to_id(self.get_id()))
        except ValueError:
            pass
        labels += [lbl_element_type, lbl_x, lbl_y, lbl_z]
        values.append(periodic_list[self.num])
        values.extend(self.coords)
        for l, v in zip(labels, values):
            element.set(l, str(v))
        return element

    def to_xyz(self, style='xyz'):
        '''
        Return a string representation of the atom suitable for an xyz file
        '''
        if style in ['xyz', 'gauss']:
            prefix = periodic_list[self.num]
        elif style == 'nw':
            prefix = periodic_list[self.num].lower()
        elif style == 'gms':
            prefix = periodic_list[self.num].ljust(10) + str(self.num) + '.0'
        else:
            raise ValueError('%s is not a valid xyz format' % style)
        return '\t'.join([prefix] + [str(i) for i in self.coords])

    def get_hybridisation(self):
        '''
        Returns the p element of an atom's hybridiation
        e.g. C_acetylene.get_hybridisation -> 1
        C_benzene.get_hybridisation -> 2
        C_methane.get_hybridisation -> 3
        Fails if d elctrons are involved
        '''
        all_ids_bonded = self.get_bond_ids()
        idx_self = self.get_id()
        bond_orders = [self.mol.get_bond_order(idx_self, each_idx_bonded) - 1
                       for each_idx_bonded in all_ids_bonded]
        return int(3 - sum(bond_orders))

    def rotate(self, matrix):
        '''
        Rotate the atom by the supplied rotation matrix
        '''
        self.coords = matrix.dot(self.coords)


def from_cml(element):
    '''
    Parses a cml atom element
    '''
    x = float(element.get(lbl_x))
    y = float(element.get(lbl_y))
    z = float(element.get(lbl_z))
    element_label = element.get(lbl_element_type)
    if element_label in periodic_table:
        num = periodic_table[element_label]['num']
    else:
        num = 0  # Dummy atom
    idx = id_to_py(element.get(lbl_id))
    return idx, atom(num, x, y, z)
