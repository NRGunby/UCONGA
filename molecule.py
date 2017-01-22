import xml.etree.ElementTree as ET
from itertools import combinations
from collections import deque
import linalg
import numpy
import atom
from math import atan2, sin, cos, degrees, acos
from constants import *
import warnings


class molecule(object):
    '''
    Represents a molecule
    Some of the design is inspired by OpenBabel's OBMol
    '''
    def __init__(self, atoms, bonds=False):
        '''
        Creates a molecule object
        Accepts:
            A list of atoms
            Optionally: A bond array, where the ijth entry is the bond order
                        between atoms i and j
        Returns:
            A molecule object
        '''
        self.atoms = atoms
        self.aromatised_bonds = []
        for at in self.atoms:
            at.mol = self
        if bonds:
            self.bonds = numpy.array(bonds, numpy.dtype(float))
        else:
            self.bonds = numpy.array([[0 for j in self.atoms] for i in self.atoms],
                                     numpy.dtype(float))
        self.distances = self.bonds
        self.rings = []
        if bonds:
            self.update()

    def update(self):
        '''
        Groups all steps that must be done after adding atom(s) or bond(s) together
        '''
        self.build_distances()
        self.find_rings()
        self.label_aromatics()


    def is_valid_id(self, idx):
        '''
        Checks that something is the id of one of this molecule's atoms
        Accetps:
            An id
        Returns:
            A boolean
        '''
        return idx >= 0 and idx < len(self.atoms) and isinstance(idx, (int, long))

    def add_atom(self, new_atom, done=True):
        '''
        Adds an atom
        Accepts:
            An atom object
            Optional: A boolean representing whether to rebuild other arrays or not
                      This defaults to True, but if many atoms are being added set to
                      False for all but the last to increase efficiency
        '''
        self.atoms.append(new_atom)
        new_atom.mol = self
        new_num_atoms = len(self.atoms)
        new_array = numpy.zeros((new_num_atoms, new_num_atoms))
        for i in range(new_num_atoms - 1):
            for j in range(new_num_atoms - 1):
                new_array[i][j] = self.bonds[i][j]
        self.bonds = new_array
        if done:
            self.update()

    def add_bond(self, id1, id2, order, done=True):
        '''
        Adds a bond
        Accepts:
            The int ids of the two atoms to bond
            The int bond order
            Optional: A boolean representing whether to rebuild other arrays or not
                      This defaults to True, but if many atoms are being added set to
                      False for all but the last to increase efficiency
        '''
        if id1 == id2:
            raise(ValueError, 'Cannot bond an atom to itself')
        elif not (self.is_valid_id(id1) and self.is_valid_id(id2)):
            raise(ValueError, 'Invalid atom id')
        else:
            self.bonds[id1][id2] = order
            self.bonds[id2][id1] = order
            self.atoms[id1].ids_cache = []
            self.atoms[id2].ids_cache = []
            if done:
                self.update()

    def all_bonds(self):
        '''
        Iterate over all bonds
        Yields pairs of atom ids
        '''
        for idx_atom_1, idx_atom_2 in combinations(range(len(self.atoms)), 2):
            if self.bonds[idx_atom_1][idx_atom_2]:
                yield idx_atom_1, idx_atom_2

    def get_bond_order(self, id1, id2):
        '''
        Returns the bond order (1, 2, etc) between two atoms
        If they are not bonded, returns 0
        If they are part of an aromatic ring, returns 1.5
        Accepts:
            Two int atom ids
        Returns:
            An int
        '''
        return self.bonds[id1][id2]

    def get_angle(self, id_1, id_center, id_2):
        '''
        Returns the angle in radians defined by three atoms
        Accepts:
            Three atom ids, where the middle id is the center of the angle
        Returns:
            A float
        '''
        atoms = [self.atoms[i] for i in id_1, id_center, id_2]
        a, b = [atoms[1].get_vector(i) for i in (atoms[0], atoms[2])]
        tmp = a.dot(b)/(numpy.linalg.norm(a)*numpy.linalg.norm(b))
        # Deal with possible floating point rounding problems
        if tmp > 1:
            tmp = 1
        if tmp < -1:
            tmp = -1
        return acos(tmp)

    def coord_matrix(self):
        '''
        Returns a numpy array of the coordinates of all the atoms in the molecule
        '''
        return numpy.array([i.coords for i in self.atoms])

    def all_torsions(self):
        '''
        Iterate over all torsions
        Yields 4-tuples of atom ids
        '''
        for middle in self.all_bonds():
            ends = [self.atoms[each].get_bond_ids() for each in middle]
            for each in ends:
                for m in middle:
                    if m in each:
                        each.remove(m)
            if ends[0] and ends[1]:
                caps = [i[0] for i in ends]
                caps[1:1] = middle
                yield tuple(caps)

    def all_pairs(self):
        '''
        Iterate over all nonbonded pairs
        '''
        for idx_atom_1, idx_atom_2 in combinations(range(len(self.atoms)), 2):
            if self.distances[idx_atom_1][idx_atom_2] > 2:
                yield idx_atom_1, idx_atom_2

    def build_distances(self):
        '''
        Rebuilds the list of (graph-theoretical, not structural) distances between atoms
        '''
        # Algorithm details:
        # This relies on taking consecutive powers of the adjacency matrix
        # (here the bond matrix)
        # If the ijth entry in the nth power is non-0, then there is an
        # n-length path between atoms i and j
        # Inefficient, but the time taken is trivial compared with
        # the conformer generation and testing.
        unfound_atom_pairs = set([i for i in combinations(range(len(self.atoms)), 2)])
        current_distance = 1
        working_matrix = numpy.array([[j for j in i] for i in self.bonds])
        distances_matrix = numpy.array([[0 for j in i] for i in self.bonds])
        limit = len(self.atoms)
        while unfound_atom_pairs:
            for each_atom_idx_pair in set(unfound_atom_pairs):
                idx_atom_1, idx_atom_2 = each_atom_idx_pair
                # Is there a path between these atoms,
                # and have they not been connected before?
                if (working_matrix[idx_atom_1][idx_atom_2] and not
                        distances_matrix[idx_atom_1][idx_atom_2]):
                    distances_matrix[idx_atom_1][idx_atom_2] = current_distance
                    distances_matrix[idx_atom_2][idx_atom_1] = current_distance
                    unfound_atom_pairs.remove(each_atom_idx_pair)
            current_distance += 1  # We've checked all pairs - onto the next power
            if current_distance > limit:
                warnings.warn("Some atoms unconnected. Continuing as best possible.",
                              RuntimeWarning)
                break
            working_matrix = working_matrix.dot(self.bonds)
            self.distances = distances_matrix

    def find_rings(self):
        '''
        Updates the list of rings.
        This consists of all rings, not just the smallest set of smallest rings.
        '''
        self.rings = []
        idx = 0
        queue_chains = []
        while (not len(queue_chains)) and idx < len(self.atoms):
            queue_chains = deque([[idx, i] for i in self.atoms[idx].get_bond_ids()])
            idx += 1
        # Algorithm details:
        # This looks at each 'chain' of bonded atoms, and grows it until it
        # doubles back on itself (a ring) or
        # until it reaches the edge of the molecule (not a ring)
        while queue_chains:
            current_chain = queue_chains.popleft()
            all_ids_child = self.atoms[current_chain[-1]].search_away_from(current_chain[-2])
            for each_idx_child in all_ids_child:
                if each_idx_child in current_chain: # This defines a ring
                    idx_chain_closure = current_chain.index(each_idx_child)
                    new_ring = current_chain[idx_chain_closure:]
                    # Avoid double-counting by only adding the 'clockwise' ring
                    if new_ring[1] < new_ring[-1]:
                        self.rings.append(new_ring + [each_idx_child])
                else:
                    queue_chains.append(current_chain + [each_idx_child])  # Keep searching
        # Some double-counting will have occured, so remove duplicate rings
        sorted_rings = [sorted(i[:-1]) for i in self.rings]
        idx_working = 0
        while idx_working < len(self.rings):
            if sorted_rings.index(sorted_rings[idx_working]) != idx_working:
                # There is a copy of this ring in another order further to the left
                del sorted_rings[idx_working]
                del self.rings[idx_working]
                # Do not increment, since there is a new ring at the working idx
            else:
                idx_working += 1

    def label_aromatics(self):
        '''
        Changes the order of all aromatic bonds to 1.5
        Keeps a record of what has been changed
        '''
        reference = set([frozenset([q]) for q in (1, 2)])
        for each_ring in self.rings:
            bond_orders = [self.bonds[j][k] for j, k in zip(each_ring[:-1], each_ring[1:])]
            alternating_bond_orders = (bond_orders[1::2], bond_orders[::2])
            to_test = set([frozenset(q) for q in alternating_bond_orders])
            if to_test == reference:
                for idx_at_1, idx_at_2 in zip(each_ring[:-1], each_ring[1:]):
                    self.aromatised_bonds.append((idx_at_1, idx_at_2,
                                                 self.bonds[idx_at_1][idx_at_2]))
                    self.bonds[idx_at_1][idx_at_2] = 1.5
                    self.bonds[idx_at_2][idx_at_1] = 1.5

    def is_in_ring(self, id1, id2):
        '''
        Checks if bond between two atoms with given ids is in a ring
        Accepts:
            Two atom ids
        Returns:
            A boolean
        '''
        for each_ring in self.rings:
            if id1 in each_ring and id2 in each_ring:
                return True
        return False

# A collection of utility methods refactored out of get_morgan_equivalencies
# These are all elated to nuclear permutational symmetry

    def follow_dbl_chain(self, id1):
        '''
        A utility for the other utilities
        Given an atom id, returns the other end of the double-bonded chain
        Designed for condensing allenes and other cumulenes
        Accepts:
            An atom id
        Returns:
            None if the atom called on is not the end of a double-bonded chain
            An (atom id, chain length) tuple if the atom called is on the end
        '''
        len_chain = 1
        old_id = id1
        curr_bonds = [i for i in self.bonds[old_id]]
        if curr_bonds.count(2.0) != 1:
            return None  # We came into the middle of a chain or a non-chain
        curr_id = curr_bonds.index(2.0)
        while True:
            curr_bonds = [i for i in self.bonds[curr_id]]
            num_dbls = curr_bonds.count(2.0)
            if num_dbls == 1:
                return curr_id, len_chain
            elif num_dbls != 2:
                # Weird sulfur/phosphorous/metal/etc that we can't handle
                # Either that or it was called on an atom with no double bonds
                return None
            else:
                len_chain += 1
                if curr_bonds.index(2.0) != old_id:
                    # Either old_id is None or the new end is to the left of the old end
                    # Either way, the simple way is what we want
                    old_id = curr_id
                    curr_id = curr_bonds.index(2.0)
                else:
                    # The new way is to the right of the old way
                    tmp = old_id
                    old_id = curr_id
                    curr_id = curr_bonds.index(2.0, tmp + 1)

    def find_dbl_systems(self, sym_classes):
        '''
        Finds doubly-bonded systems with defined stereochemistry
        This excludes those with no stereochemistry, and those
        where (due to one end being >2-coordinate) the stereochemistry is complicated
        Accepts:
            A list of the symmetry classes of each atom in the molecule
        Yields:
            4-tuples consisting of: the number of bonds in the chain,
                                    a list of the atom ids of the
                                          substituents of the ends of the chain,
                                          the id of one end of the chain,
                                          the id of the other end of the chain
        '''
        for each_idx_1, each_bond_list in enumerate(self.bonds):
            dc = self.follow_dbl_chain(each_idx_1)
            if dc is not None:
                each_idx_2, n_bonds_in_chain = dc
                atoms = []
                for i in (each_idx_1, each_idx_2):
                    atoms.append((self.atoms[i], list(self.bonds[i]).index(2.0)))
                ends = [p[0].search_away_from(p[1]) for p in atoms]
                lengths = [len(k) for k in ends]
                if lengths[0] not it (1, 2) or lengths[1] not in (1, 2):
                    # Either this is a carbonyl etc with no stereochemistry or
                    # this is not an organic double bond as one end has
                    # too many substituents so
                    # Stereochemistry here is not well-definded
                    pass
                else:
                    if each_idx_1 < each_idx_2:
                        should_yield = True
                        for j in ends:
                            if len(set([sym_classes[i] for i in j])) != len(j):
                                should_yield = False
                        if should_yield:
                            yield n_bonds_in_chain, ends, each_idx_1, each_idx_2

    def assign_dbl_stereochem(self, sym_classes):
        '''
        Finds and assigns stereocehmistry to double-bond collections,
        namely alkenes, allenes, and higher cumulenes.
        Accepts:
            A list of the symmetry classes of all atoms in the molecule
            Returns: A list of the double-bond-group stereochemistry-type
                     of all atoms in the molecule. This is 0 if there is no
                     double-bond-group stereochemistry, and 1 or 2 arbritratily but
                     consistently otherwise
        '''
        double_stereochemistry = [0 for i in self.atoms]
        for n_bonds_in_chain, ends, each_idx_1, each_idx_2 in self.find_dbl_systems(sym_classes):
            if double_stereochemistry[each_idx_1] == double_stereochemistry[each_idx_2] == 0:
                # Choose the appropriate trigonometric function for the geometry
                # I.e. the one that will give values of ~ 1 and ~ -1
                # Cosine for coplanar, sine for perpendicular
                if n_bonds_in_chain % 2:
                    op = cos
                else:
                    op = sin
                sorted_ends = [sorted(i, key=lambda x: sym_classes[x])
                               for i in ends]
                torsion = self.get_torsion(sorted_ends[0][-1], each_idx_1,
                                           each_idx_2, sorted_ends[1][-1])
                if op(torsion) > 0:
                    double_stereochemistry[each_idx_1] = 2
                    double_stereochemistry[each_idx_2] = 2
                else:
                    double_stereochemistry[each_idx_1] = 1
                    double_stereochemistry[each_idx_2] = 1
        return double_stereochemistry

    def assign_tet_stereochem(self, new_ranks):
        '''
        Another utility for get_morgan_equivalencies
        Same logic as assign_dbl_stereochem
        Stereochemistry is determined by seeing if the 2nd-
        highest priority substituent is clockwise or
        anticlockwise of the highest-priority substituent
        '''
        tetrahedral_stereochemistry = [0 for i in self.atoms]
        for each_idx, each_atom in enumerate(self.atoms):
            all_ids_neighbours = each_atom.get_bond_ids()
            # An atom is a chiral centre if it has four neighbours, none of which
            # are in the same symmetry class
            num_neighbours = len(all_ids_neighbours)
            num_unique_neighbours = len(set([new_ranks[q] for q in all_ids_neighbours]))
            if num_neighbours == num_unique_neighbours == 4:
                sorted_ids_neighbours = sorted(all_ids_neighbours, key=lambda x: new_ranks[x])
                # Center on the atom by using the vectors between the neighbours and the
                # central atom instead of using the neighbours' coordinates
                # Transpose the vectors for ease of multiplication later
                vectors = [each_atom.get_vector(self.atoms[j]).transpose()
                           for j in sorted_ids_neighbours]
                # Align the lowest-priority substitutent with the z axis:
                low_z_mat = linalg.rotation_from_axes(vectors[0].transpose(), [0, 0, 1])
                z_aligned_vectors = [low_z_mat.dot(q) for q in vectors]
                # Align the highest-priority substituent with the x axis:
                high_x_mat = linalg.rotation_from_axes(vectors[-1].transpose(), [1, 0, 0])
                aligned_vectors = [high_x_mat.dot(q) for q in z_aligned_vectors]
                number_2 = aligned_vectors[-2].transpose()
                if atan2(number_2[1], number_2[0]) > 0:
                    tetrahedral_stereochemistry[each_idx] = 2
                else:
                    tetrahedral_stereochemistry[each_idx] = 1
        return tetrahedral_stereochemistry

    def find_para_candidates(self, sym_classes):
        '''
        Para-stereocentres, as defined in Razinger et al,
        J. Chem. Inf. Comput. Sci., 1993, p 812
        A para-stereocentre is a tetrahedral atom with three different
        groups or a double bond with different groups on one end and
        identical groups on the other, where the identical groups are
        part of a ring assembly (in the same ring , or different rings
        connected via double bonds or spiro atoms) with other para-stereocentres
        This captures in-ring cis-trans isomerisation (eg 1,3-
        dimethylcyclobutane) and a lot more
        '''
        all_ids_ring_atoms = []
        for i in self.rings:
            # Ignore the repetition of the ring closure
            all_ids_ring_atoms += i[:-1]
        # If an atom is in more than one ring, it will be on the list multiple times
        # This is important for finding extended ring systems
        # Start by finding candidate parastereocentres
        candidates = []
        for each_idx, each_atom in enumerate(self.atoms):
            if each_idx in all_ids_ring_atoms:
                neighbours = each_atom.get_bond_ids()
                ring_neighbours = [j for j in neighbours if self.is_in_ring(each_idx, j)]
                other_neighbours = [j for j in neighbours if j not in ring_neighbours]
                # If the in-ring neighbours are different, then atom may be a
                # true stereocentre, but not a para-stereocentre
                if (len(set([sym_classes[x] for x in ring_neighbours]))) == 1:
                    # Note: this does not find doubly-bonded parastereocentres
                    # It find the in-ring atoms they are bonded to.
                    # The actual doubly-bonded stereocentres are found through the
                    # later attempts to connect everything
                    end_sym_classes = [sym_classes[i] for i in
                                       self.atoms[self.follow_dbl_chain(each_idx)[0]].get_bond_ids()]
                    dbl_para = (len(other_neighbours) == 1 and
                                self.bonds[each_idx][other_neighbours[0]] == 2 and
                                len(set(end_sym_classes)) == len(end_sym_classes) and
                                len(end_sym_classes) > 1)
                    tetrahedral_para = (len(set([sym_classes[x] for x in
                                                 other_neighbours])) == 2)
                    if dbl_para or tetrahedral_para:
                        candidates.append(each_atom)
        return all_ids_ring_atoms, candidates

    def find_para_groups(self, sym_classes):
        '''
        Groups cantidate parastereocentres by which ring system they are in
        The ring systems returned by ring_lib can't be used since here we consider rings joined by
        spiro centres and double bonds to be part of the same system
        '''
        all_ids_ring_atoms, candidates = self.find_para_candidates(sym_classes)
        para_groups = []
        paras_touched = []
        # Group the parastereocentres together
        for each_atom_candidate in candidates:
            if each_atom_candidate not in paras_touched:
                paras_touched.append(each_atom_candidate)
                i_groups = [each_atom_candidate]
                rings_stack = []
                # Start by finding the ring(s) that this atom is in
                for r in self.rings:
                    if each_atom_candidate.get_id() in r:
                        rings_stack.append(r)
                curr_ring_idx = 0
                while curr_ring_idx < len(rings_stack):
                # Then find the rings connected to that ring
                # And the parastereocentre candidates in all of them
                    curr_ring = rings_stack[curr_ring_idx]
                    for at_idx in curr_ring:  # Is the system extended?
                        if all_ids_ring_atoms.count(at_idx) > 1:
                            # The atom is in a second ring
                            for r in self.rings:
                                # If we haven't already looked through this ring
                                # add it to the stack
                                if at_idx in r and r not in rings_stack:
                                    rings_stack.append(r)
                        if 2 in self.bonds[at_idx]:
                            # The atom is double-bonded - look at the other end
                            other_idx = list(self.bonds[at_idx]).index(2)
                            if other_idx in all_ids_ring_atoms:  # Extend the system
                                for r in self.rings:
                                    if other_idx in r and r not in rings_stack:
                                        rings_stack.append(r)
                    curr_ring_idx += 1
                for other_at_candidate in candidates:
                    idx_other_candidate = other_at_candidate.get_id()
                    if (other_at_candidate != each_atom_candidate and
                       True in [idx_other_candidate in i for i in rings_stack] and
                       other_at_candidate not in paras_touched):
                        i_groups.append(other_at_candidate)
                        paras_touched.append(other_at_candidate)
                # That's all the conncted atoms found
                if len(i_groups) > 1:
                    # Replace any double-bonded atoms with
                    # whatever they are double-bonded to (the real parastereocentre):
                    for idx_in_group, at in enumerate(i_groups):
                        idx_in_molecule = at.get_id()
                        if 2 in self.bonds[idx_in_molecule]:
                            other_idx = self.follow_dbl_chain(idx_in_molecule)[0]
                            i_groups[idx_in_group] = self.atoms[other_idx]
                    para_groups.append(deque(i_groups))
        return para_groups

    def assign_para_stereochem(self, sym_classes):
        '''
        Assigns labels to the parastereocentres previously found and grouped
        Those that are in a group of one can be discarded,
        as they are not actually parastereocentres.
        Those in larger groups have labels assigned based on their relative
        stereochemistry to the other parastereocentres in the group
        '''
        para_stereochemistry = [0 for i in self.atoms]
        para_groups = self.find_para_groups(sym_classes)
        all_ids_ring_atoms, candidates = self.find_para_candidates(sym_classes)
        # Now we have all the groups of parastereocentres, time to assign the geometry
        # This is more complicated than for cis/trans in a double bond, because there
        # can be more than two in a group (e.g. 1, 3, 5-trimethylcyclohexane,
        # which can be all up or one down)
        # We choose to resolve this by, for each atom, sorting the rest of its group
        # by distance from the parent and symmetry class,
        # then building a classification out of the classifications for each neighbour
        for group in para_groups:
            l = len(group)
            for each_at_idx in range(l):
                head = group[0]
                head_id = head.get_id()
                head_hp_subs = []
                for i in head.get_bond_ids():
                    if (i not in all_ids_ring_atoms) and (self.bonds[head_id][i] == 1):
                        head_hp_subs.append(x)
                head_hp_subs.sort(key=lambda x: sym_classes[x])
                head_hp_sub = head_hp_subs[-1]
                tail = list(group)[1:]
                stereochem_list = []
                tail.sort(key=lambda x: (sym_classes[x.get_id()],
                                         self.distances[head_id][x.get_id()]))
                for tail_at in tail:
                    tail_id = tail_at.get_id()
                    tail_hp_subs = []
                    for i in tail_at.get_bond_ids():
                        if (i not in all_ids_ring_atoms) and (self.bonds[tail_id][i] == 1):
                            tail_hp_subs.append(i)
                    tail_hp_subs.sort(key=lambda x: sym_classes[x])
                    tail_hp_sub = tail_hp_subs[-1]
                    t = self.get_torsion(head_hp_sub, head_id, tail_id, tail_hp_sub)
                    res = [sin(t), cos(t)]
                    res.sort(key=lambda x: abs(x))
                    if res[-1] > 0:
                        stereochem_list.append('1')
                    else:
                        stereochem_list.append('0')
                stereo_val = int(''.join(stereochem_list), 2) + 1
                para_stereochemistry[head_id] = stereo_val
                group.rotate()
        return para_stereochemistry

    def get_morgan_equivalencies(self):
        '''
        Modified Morgan algorithm for classifying atoms by nuclear
            permutational symmetry
        Based on Moreau, Nouv. J. Chim., 1980, p 17

        My changes include:
        Not performing a charge-based step
        Modifying the double-bond step to include allenes and highere cumulenes
        Adding a step to check for parastereocentres

        Unhandled edge-cases:
            Axial chirality (e.g. DIPHOS)
            Helical chirality (e.g. Fe(acac)3)
            Octahedral diastereoisomerism (cis/trans and mer/fac)
            Extended chirality (e.g. asymmetric tetrahedranes
                                and some fullerenes)
            square-planar centres (cis/trans)
        '''
        all_ids_neighbours = [each.get_bond_ids() for each in self.atoms]
        # Basic Morgan algorithm using extended connectivites
        connectivities = [each.get_heavy_valence() for each in self.atoms]
        initial_ranks = rank(connectivities)
        new_ranks = rank_until_converged(initial_ranks, all_ids_neighbours)
        # Now the modified part starts
        # It uses the pi functionality and atomic number
        # Moreau is somewhat unclear about what pi functionality is
        # I have interpreted it to mean number of multiple bonds
        pi_functionalities = [len(filter(lambda x: x > 1, i))
                              for i in self.bonds]
        atomic_numbers = [each.num for each in self.atoms]
        for atom_property in (pi_functionalities, atomic_numbers):
            init_ranks = [(r << 8) + a for r, a in
                          zip(new_ranks, atom_property)]
            new_ranks = rank_until_converged(init_ranks, all_ids_neighbours)
        # Stereochemistry handling uses methods, not lists
        # So it needs a seperate loop.
        for stereochem_assigner in (self.assign_dbl_stereochem,
                                    self.assign_tet_stereochem,
                                    self.assign_para_stereochem):
            stereochem = stereochem_assigner(new_ranks)
            init_ranks = [(r << 8) + a for r, a in
                          zip(new_ranks, stereochem)]
            new_ranks = rank_until_converged(init_ranks, all_ids_neighbours)
        return new_ranks

    def get_torsion(self, id1, id2, id3, id4):
        '''
        Returns the 1-2-3-4 torsion in radians
        '''
        atoms = [self.atoms[i] for i in (id1, id2, id3, id4)]
        bond_vectors = [i.get_vector(j) for i, j in zip(atoms[:-1], atoms[1:])]
        c12 = numpy.cross(bond_vectors[0], bond_vectors[1])
        c23 = numpy.cross(bond_vectors[1], bond_vectors[2])
        term1 = numpy.cross(c12, c23).dot(linalg.normalise(bond_vectors[1]))
        term2 = c12.dot(c23)
        return atan2(term1, term2)

    def set_torsion(self, id1, id2, id3, id4, torsion):
        '''
        Rotates atom 1 and all its children aound that atom2-atom3 bond
        so that the 1-2-3-4 torsion is the supplied value (in radians)
        '''
        # Start by finding the children of atom 2(all closer to it than atom 3)
        children = []
        for idx, i in enumerate(self.distances):
            if i[id2] < i[id3]:
                children.append(idx)
        at2 = self.atoms[id2]
        axis = at2.get_vector(self.atoms[id3])
        angle = self.get_torsion(id1, id2, id3, id4) - torsion
        matrix = linalg.rotation_axis_angle(axis, angle)
        back_again = at2.coords
        there = [-1 * i for i in back_again]
        for at_idx in children:
            at = self.atoms[at_idx]
            at.translate(there)
            c = at.coords.transpose()
            new_c = matrix.dot(c)
            at.coords = new_c.transpose()
            at.translate(back_again)
        d = self.get_torsion(id1, id2, id3, id4)
        try:
            assert abs(sin(d) - sin(torsion)) < 1e-3
            assert abs(cos(d) - cos(torsion)) < 1e-3
        except AssertionError:
            lbl = '->'.join([py_to_id(i) for i in (id1, id2, id3, id4)])
            msg = 'Failed to set torsion % s' % lbl
            msg += '\n Should be %f, is %f' % (degrees(torsion), degrees(d))
            raise RuntimeError(msg)

    def center(self):
        '''
        Moves the centre of the atomic positions (NOT the centre of mass) to [0,0,0]
        '''
        centre = numpy.mean(self.coord_matrix(), axis=0)
        to_centre = -1*centre
        self.translate(to_centre)

    def translate(self, vector):
        '''
        Translates the entire molecule by the supplied 3-vector
        '''
        for i in self.atoms:
            i.translate(vector)

    def rotate(self, matrix):
        '''
        Rotates the entire molecule by the supplied rotation matrix
        '''
        for i in self.atoms:
            i.rotate(matrix)

    def copy(self):
        '''
        Returns a copy of the molecule
        '''
        new_atoms = [each.copy() for each in self.atoms]
        new_bonds = [[j for j in i] for i in self.bonds]
        new_mol = molecule(new_atoms, new_bonds)
        new_mol.aromatised_bonds = self.aromatised_bonds
        return new_mol

    def copy_subset(self, ids):
        '''
        Generic method to get a fragment of a molecule
        Can also be used to renumber the atoms in a molecule
        '''
        new_atoms = [self.atoms[each].copy() for each in ids]
        new_bonds = [[self.bonds[j][i] for j in ids] for i in ids]
        trans_dict = {i: idx for idx, i in enumerate(ids)}
        new_mol = molecule(new_atoms, new_bonds)
        trans_arom_bds = []
        for i in self.aromatised_bonds:
            if i[1] in trans_dict and i[0] in trans_dict:
                trans_arom_bds.append((trans_dict[i[0]], trans_dict[i[1]], i[2]))
        new_mol.aromatised_bonds.extend(trans_arom_bds)
        return new_mol

    def copy_without_H(self):
        '''
        Removes stereochemically unimportant hydrogen atoms.
        A hydrogen is stereochemically unimportant if it is a) not bridging and
        b) not the only hydrogen attached to an atom
        Returns:
            A molecule object
            An old id -> new id translation table
        '''
        hless_new_atoms = []
        trans = [-1]*len(self.atoms)
        for idx, at in enumerate(self.atoms):
            if at.num != 1:
                hless_new_atoms.append(at.copy())
                trans[idx] = len(hless_new_atoms) - 1
            else:
                bonded = at.get_bond_ids()
                if len(bonded) > 1:  # Bridging
                    hless_new_atoms.append(at.copy())
                    trans[idx] = len(hless_new_atoms)
                elif (self.atoms[bonded[0]].get_heavy_valence() + 1 ==
                      self.atoms[bonded[0]].get_bond_ids()):  # The only h atom
                    hless_new_atoms.append(at.copy())
                    trans[idx] = len(hless_new_atoms)
        new_mol = molecule(hless_new_atoms)
        for each_idx_row, each_row in enumerate(self.bonds):
            for each_val_idx, each_val in enumerate(each_row):
                new = [trans[each_idx_row], trans[each_val_idx], each_val, False]
                if -1 not in new and each_val > 0:
                    new_mol.add_bond(*new)
        new_mol.update()
        trans_dict = {i: idx for idx, i in enumerate(trans) if i > -1}
        trans_arom_bds = []
        for i in self.aromatised_bonds:
            if i[1] in trans_dict and i[0] in trans_dict:
                trans_arom_bds.append((trans_dict[i[0]], trans_dict[i[1]], i[2]))
        new_mol.aromatised_bonds.extend(trans_arom_bds)
        return new_mol, trans_dict

    def test_ordering(self, ordering):
        '''
        Test if a permutation is an automorphism
        See Razinger et al, J. Chem. Inf. Comput. Sci. 1993 (33), 197-201
        '''
        permutation = [[0 for j in self.atoms] for i in self.atoms]
        for idx, i in enumerate(ordering):
            permutation[idx][i] = 1
        permutation_matrix = numpy.array(permutation)
        inverse_permutation = permutation_matrix.transpose()
        partial_product = self.bonds.dot(permutation_matrix)
        product_to_test = inverse_permutation.dot(partial_product)
        # Needs replacing by approximation
        if numpy.equal(product_to_test, self.bonds).all():
            return True
        else:
            return False

    def to_cml(self):
        '''
        Creates a cml molecule element
        Uses the original (i.e. non-aromatised)
        '''
        original_bonds = self.bonds
        for each_tuple in self.aromatised_bonds:
            i, j, k = each_tuple
            original_bonds[i][j] = k
            original_bonds[j][i] = k
        root = ET.Element(lbl_molecule)
        atom_array = ET.Element(lbl_atom_array)
        for each in self.atoms:
            atom_array.append(each.to_cml())
        root.append(atom_array)
        bond_array = ET.Element(lbl_bond_array)
        for i_idx, i in enumerate(original_bonds):
            for j_idx, j in enumerate(i):
                if j and (i_idx > j_idx):
                    bond = ET.Element(lbl_bond)
                    bond.set(lbl_order, str(j))
                    bond.set(lbl_atom_refs,
                             ' '.join([py_to_id(l) for l in [i_idx, j_idx]]))
                    bond_array.append(bond)
        root.append(bond_array)
        return ET.tostring(root)

    def to_xyz(self, format='xyz'):
        '''
        Returns the text of an xyz file describing the molecule
        '''
        ret = []
        if format == 'xyz':
            ret.append(str(len(self.atoms)))
            ret.append('Generated by UCONGA')
        elif format == 'gauss':
            ret.append('Generated by UCONGA')
            ret.append('')
            ret.append('0 1')
        elif format == 'nw':
            pass
        elif format == 'gms':
            ret.append('Generated by UCONGA')
            ret.append('C1')
        for i in self.atoms:
            ret.append(i.to_xyz(format))
        return '\n'.join(ret)


def from_cml(file_name):
    '''
    Parses a cml file
    '''
    try:
        tree = ET.parse(file_name)
        root = tree.getroot()
        atom_array = [i for i in root if lbl_atom_array in i.tag][0]
        b_atoms = [atom.from_cml(i) for i in atom_array if lbl_atom in i.tag]
        b_atoms.sort()  # Guarantee that the ids will be correct
        atoms = [i[1] for i in b_atoms]
        m = molecule(atoms)
        bond_array = [i for i in root if lbl_bond_array in i.tag][0]
        for bond in [i for i in bond_array if lbl_bond in i.tag]:
            text = bond.get(lbl_atom_refs)
            ids = [id_to_py(i) for i in text.split()]
            order = float(bond.get(lbl_order))
            m.add_bond(*ids, order=order, done=False)
        m.update()
        return m
    except AttributeError:
        msg = ('%s is not a valid cml-formatted molecule. Stopping.' % file_name)
        raise RuntimeError(msg)

# Utility methods for get_Morgan_equivalencies that aren't molecule methods


def rank(weights):
    '''
    A utility method to perform the ranking step
    in a Morgan-based algorithm
    '''
    working_list = sorted([[i, idx] for idx, i in enumerate(weights)])
    sorted_weights = sorted(weights)
    adder = [sorted_weights[0:idx].count(i)
             for idx, i in enumerate(sorted_weights)]
    working_list_2 = [[i[1], idx + 1]
                      for idx, i in enumerate(working_list)]
    for weight_tuple, to_add in zip(working_list_2, adder):
        weight_tuple[1] = weight_tuple[1] - to_add
    # Return to original order and extract the ranks
    ret = [i[1] for i in sorted(working_list_2)]
    return ret


def rank_until_converged(initial_ranks, all_ids_neighbours):
    '''
    Updates the ranking in a Morgan algorithm
    '''
    old_ranks = []
    new_ranks = initial_ranks
    iteration_count = 0
    while new_ranks != old_ranks:
        weights = [(((len(j) + 1) * i) + sum([new_ranks[k] for k in j]))
                   for i, j in zip(new_ranks, all_ids_neighbours)]
        old_ranks = new_ranks
        new_ranks = rank(weights)
        iteration_count += 1
        if iteration_count > 1000:
            msg = 'Symmetry-detection algorithm did not converge in 1000 iterations.'
            raise RuntimeError(msg)
    return new_ranks
