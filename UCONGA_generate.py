from math import radians, degrees, sin, cos
import molecule
import ring_lib
import numpy
import bisect
import warnings
from itertools import chain, cycle, product
import atom
linear_tolerance = radians(170)


def choose_scaling(mol):
   '''
   Selects a van der Waals scaling factor based on the steric crowding
   An infinite linear alkane (average heavy valence=2) will have a high scaling factor
   An infinite diamondoid (average heavy valence=4) will have a low scaling factor
   '''
   count = 0.0
   tally = 0.0
   for each_atom in mol.atoms:
      hvy_val = each_atom.get_heavy_valence()
      if hvy_val > 1:
         count += 1
         tally += hvy_val
   average_steric = tally/count
   strict_scaling = 0.9
   loose_scaling = 0.7
   scaling_delta = strict_scaling - loose_scaling
   high_steric = 4.0 # Diamandoid
   low_steric = 2.0 # Alkane
   steric_delta = high_steric - low_steric
   steric_percent = (average_steric - low_steric)/steric_delta
   scaling_percent = 1.0 - steric_percent # High steric crowding => low cutoff
   scaling = loose_scaling + (scaling_percent * scaling_delta)
   return scaling

def test_pair(pair, scaling):
    '''
        Test whether a pair of atoms are too close
        Too close means they are closer than the scaled sum of their vdW radii
        Returns True if the pair is OK and False if it is not
        '''
    radii = [each.get_vdw() for each in pair]
    if 0.7 < scaling and set([i.num for i in pair]) in [set([1, 7]), set([1, 7])]:
        scaling = 0.7 # There is the possibility of hydrogen bonding
    cutoff = scaling * sum(radii)
    distance = pair[0].get_distance(pair[1])
    if cutoff > distance:
        return False
    else:
        return True


def test_mol(mol, scaling):
    '''
    Test whether any >1,4 neighbours in an molecule are too close together
    Too close is defined as it is for test_pair
    Return True if the conformer is OK and False if it is not
    '''
    for each_pair in mol.all_pairs():
        atom_pair = [mol.atoms[each] for each in each_pair]
        # Test if the distance between the pair can be changed by rotation
        direct_path = []
        working_atom = atom_pair[0]
        end_id = each_pair[1]
        neighbour_id = sorted(working_atom.get_bond_ids(), key = lambda x: mol.distances[end_id][x])[0]
        while neighbour_id != end_id:
            direct_path.append(neighbour_id)
            working_atom = mol.atoms[neighbour_id]
            neighbour_id = sorted(working_atom.get_bond_ids(), key = lambda x: mol.distances[end_id][x])[0]
        can_change = True
        for each_bond in zip(direct_path[:-1], direct_path[1:]):
            if not test_rotatable(*[mol.atoms[i] for i in each_bond]):
                can_change = False
        if can_change and test_pair(atom_pair, scaling) is False:
            return False
    else: # Not technically necessary but clearer
        return True


def test_ring(atom_1, atom_2):
    '''
    Test if the bond between two atoms is prevented
    from rotation due to being in a ring
    Returns True if not in a ring, False if it is
    '''
    ids = [each.get_id() for each in (atom_1, atom_2)]
    return not atom_1.mol.is_in_ring(*ids)



def test_order(atom_1, atom_2):
    '''
    Test if the bond between two atoms is prevented from
    rotation due to its order
    Returns True for single bonds, False for multiple bonds
    '''
    ids = [each.get_id() for each in (atom_1, atom_2)]
    return atom_1.mol.get_bond_order(*ids) < 1.5


def test_interesting(atom_1, atom_2):
    '''
    Tests if there is any point to rotating about the bond
    Returns True if each end has at least one heavy atom (defined as > H)
    and, if there is only one, it is not linear
    '''
    ats = [atom_1, atom_2]
    this_mol = atom_1.mol
    heavy_neighbours = [[j for j in each.search_away_from(other.get_id()) if this_mol.atoms[j].num != 1]
                        for each, other in zip(ats, ats[::-1])]
    # Do both ends have heavy neighbours?
    if 0 in [len(i) for i in heavy_neighbours]:
        return False
    # Is either end nearly linear?
    elif True in [len(i) == 1 and abs(this_mol.get_angle(i[0], each.get_id(), other.get_id())) > linear_tolerance
                  for i, each, other in zip(heavy_neighbours, ats, ats[::-1])]:
        return False
    else:
        return True


def test_rotatable(atom_1, atom_2):
    '''
    Tests if the bond between two atoms is rotatable
    A bond is rotatable if it is single, not in a ring,
    and has at least one heavy atom attached to each end
    '''
    return (test_interesting(atom_1, atom_2) and
            test_ring(atom_1, atom_2) and test_order(atom_1, atom_2))


def find_rotatable_bonds(mol):
    '''
    Finds all rotatable bonds in a molecule
    Rotatable bonds are single bonds not in a ring that have at least one
    non-H atom at each end
    '''
    torsions = [[i for i in each]
                for each in mol.all_torsions()]
    results = []
    for each_torsion in torsions:
        atoms = [mol.atoms[each_atom] for each_atom in each_torsion]
        bond_atoms = atoms[1:3]
        if test_rotatable(*bond_atoms):
            results.append(tuple([each.get_id() for each in atoms]))
    return results


def make_rotamer(mol, rotatable_bonds, new_values):
    '''
    Copies a molecule and increments torsions of rotatable bonds
    '''
    new_rotamer = mol.copy()
    current_torsions = [new_rotamer.get_torsion(*each)
                        for each in rotatable_bonds]
    for each_torsion, each_value in zip(rotatable_bonds,
                                                    new_values):
        new_rotamer.set_torsion(each_torsion[0], each_torsion[1],
                                 each_torsion[2], each_torsion[3], each_value)
    return new_rotamer

def is_symmetrical_rotor(each_bond, backbone, classes):
    '''
    For an atom to be a symmetrical rotor, all children must be in the same symmetry class
    All children must be childless or in a ring rith the atom
    All children must be equidistant in a Newmann pprojection
    '''
    C_children = backbone.atoms[each_bond[2]].search_away_from(each_bond[1])
    sym_rotor = False
    if len(C_children) > 1 and len(set([classes[i] for i in C_children])) == 1 and backbone.atoms[C_children[0]].num > 0:
        all_D_children = backbone.atoms[each_bond[3]].get_bond_ids()
        if len(all_D_children) == 1 or False not in [backbone.is_in_ring(i, each_bond[3]) for i in all_D_children]:
            sym_rotor = True
            # For a n-sided regular polygon inscribed in a unit circle, for any vertex
            # the product of the distances to all other vertices is n
            # Convert to torsion angles to put everything on the unit circle
            C_child_torsions = [backbone.get_torsion(each_bond[0], each_bond[1], each_bond[2], i)
                                for i in C_children]
            C_child_coords = [numpy.array([sin(i), cos(i)]) for i in C_child_torsions]
            upper_limit = len(C_child_coords) * 1.1
            lower_limit = len(C_child_coords) * 0.9
            for i in C_child_coords:
                base_dist = 1
                for j in C_child_coords:
                    tmp = numpy.linalg.norm(i - j)
                    if tmp != 0:
                        base_dist *= numpy.linalg.norm(i - j)
                if base_dist < lower_limit or base_dist > upper_limit:
                    sym_rotor = False
    return sym_rotor


def find_older_sibling_ids(each_bond, each_id, backbone, classes, bonds_by_B_atom = {}):
    '''
    Checks if a bond is part of a group of identical bonds
    and returns the ids of all identical bonds with lower ids
    '''
    ret = []
    if each_bond[1] in bonds_by_B_atom:
        sym_classes = [classes[i] for i in each_bond]
        related_bonds = filter(lambda x: [classes[i] for i in x[1]] == sym_classes,
                               bonds_by_B_atom[each_bond[1]])
        ret.extend([str(i[0]) for i in related_bonds])
    try:
        bonds_by_B_atom[each_bond[1]].append((each_id, each_bond))
    except KeyError:
        bonds_by_B_atom[each_bond[1]] = [(each_id, each_bond)]
    return ret


def find_max_angles(rotatable_bonds, centralness, backbone, classes, allow_inversion):
    '''
    Assigns maximum torsion angles to all rotatable bonds in a system
    These default to being 360 degrees
    If the bond is to a symmetric rotor,
        the maximum is devided by the symmetry of the rotor
    If the bond is the first bond in the system and enantiomeric
        conformers are treated as identical, the maximum is divided by two
    If the bond is part of a group of identical bonds, it is limited
        to being less than the torsion angle of the bond
        immediately before it in the group
    This avoids generating any equivalent/degenerate conformers
    '''
    # Variable names are based on torsions being (centre out) A-B-C-D
    maxes = []
    # For each subunit in divide-and-conquer, we need a new dict
    curr_dict = {}
    for idx, each_bond in enumerate(rotatable_bonds):
        max = []
        # Make sure the bond is going from the centre to the periphery
        if centralness[each_bond[0]] > centralness[each_bond[-1]]:
            each_bond = each_bond[::-1]
        C_children = backbone.atoms[each_bond[2]].search_away_from(each_bond[1])
        max.extend(find_older_sibling_ids(each_bond, idx, backbone, classes, curr_dict))
        if is_symmetrical_rotor(each_bond, backbone, classes):
            max.append(360/len(C_children))
        if not max:
            max.append(360)
        if len(maxes) == 0 and allow_inversion:
            max = [i/2 if type(i) != str else i for i in max]
        maxes.append(max)
    return maxes

def make_all_rotations(mol, final_delta, scaling, allow_inversion, fix=[], vary_rings = 2):
    '''
    A generator that yields all valid rotamers of a molecule
    Scaling is a scaling factor as defined for test_pair and test_mol
    Delta is the amount (in degrees) to increment torsions by in-between trials
    Fix is a list of bonds to hold constant (important for divide-and-conquer, may be useful for custom work)
    Vary_rings can be 0 (hold all rings constant), 1 (reflect only), or 2 (full flip-of-fragments) (important for divide-and-conquer, may be useful for custom work)
    '''
    backbone, backbone_to_mol = mol.copy_without_H()
    mol_to_backbone = {v:k for k, v in backbone_to_mol.items()}
    rotatable_bonds = find_rotatable_bonds(backbone)
    for each_bond in fix:
        backbone_each_bond = sorted([mol_to_backbone[i] for i in each_bond])
        keep_looping = True
        while keep_looping:
            for i in rotatable_bonds:
                if sorted(i[1:3]) == backbone_each_bond:
                    rotatable_bonds.remove(i)
                    break
            else:
                keep_looping = False
    # Sort bonds by centralness
    centralness = [sum(i) for i in backbone.distances]
    classes = backbone.get_morgan_equivalencies()
    rotatable_bonds.sort(key=lambda x: sum([centralness[i] for i in x]))
    mol_rotatable_bonds = [[backbone_to_mol[i] for i in each]
                       for each in rotatable_bonds]
    # Build list by rules
    # Atom labels (from centre to periphery): A-B-C-D
    maxes = find_max_angles(rotatable_bonds, centralness, backbone, classes, allow_inversion)
    curr_delta = 60
    max_idx = len(rotatable_bonds) - 1
    for each_set_of_rings in ring_lib.all_ring_conformers(mol, vary_rings):
        if max_idx < 0:
            if test_mol(each_set_of_rings, scaling):
                yield each_set_of_rings
        else:
            curr_angles = [0 for i in rotatable_bonds]
            ref_conformers = numpy.array([[]])
            ref_deltas = numpy.array([[]])
            while curr_delta >= final_delta:
                idx = 0
                max_angles = [[curr_angles[int(t)] if type(t) == str else t for t in i]for i in maxes]
                possible_angles = [curr_delta * each for each in range(int(360.0/curr_delta))]
                all_possible_angles = [possible_angles[bisect.bisect_left(possible_angles, min(i))::-1]
                                       for i in max_angles]
                while idx > -1:
                    if not all_possible_angles[idx]:
                        idx -= 1
                    else:
                        curr_angles[idx] = all_possible_angles[idx].pop()
                        if idx == max_idx:
                            test_curr_angles = numpy.array(curr_angles)
                            previous_delta = curr_delta * 2
                            if numpy.mod(test_curr_angles, previous_delta).any():
                                # Check there is at least one multiple of the new angle
                                # In a modulo distance with radix r, to see if a and b are closer than d, use the formula:
                                # |(r/2) - | a - b | | > ( (r/2) - d)
                                # Here r = 2*pi and d is in the ref_deltas column vector
                                if not(len(ref_deltas[0])) or numpy.less(numpy.abs(180 - numpy.abs(ref_conformers - test_curr_angles)), 180 - ref_deltas).any(axis=1).all():
                                    with warnings.catch_warnings() as w:
                                        # Any warning messages will come from weird molecule structures
                                        # They will already have been seen in
                                        curr_rotamer = make_rotamer(each_set_of_rings,
                                                                    mol_rotatable_bonds,
                                                                    [radians(i) for i in curr_angles])
                                    if test_mol(curr_rotamer, scaling):
                                        yield curr_rotamer
                                        if not(len(ref_deltas[0])):
                                            ref_deltas = numpy.array([[curr_delta/2]])
                                            ref_conformers = numpy.array([curr_angles])
                                        else:
                                            ref_deltas = numpy.concatenate((ref_deltas, numpy.array([[curr_delta/2]])))
                                            ref_conformers = numpy.concatenate((ref_conformers, numpy.array([curr_angles])))

                        else:
                            max_angles[idx + 1] = [curr_angles[int(t)] if type(t) == str else t
                                                   for t in maxes[idx + 1]]
                            all_possible_angles[idx+1] = possible_angles[bisect.bisect_left(possible_angles,
                                                                                            min(max_angles[idx + 1]))::-1]
                            idx += 1
                curr_delta /= 2

### Divide-and-conquer functions
def divide_linear(mol, split):
    # Start by splitting the molecule into groups by breaking the bonds
    repair = []
    for each_pair in split:
        i, j = each_pair
        repair.append([i, j, mol.bonds[i][j]])
        mol.bonds[i][j] = 0
        mol.bonds[j][i] = 0
        mol.update()
    # Now find the subgroups
    group_ids = set([tuple(sorted([i] + mol.atoms[i].all_neighbours_away_from())) for i in chain(*split)])
    # Repair the main mol
    for i in repair:
        mol.bonds[i[0]][i[1]] = i[2]
        mol.bonds[i[1]][i[0]] = i[2]
        mol.update()
    return group_ids

def group_rotatable_bonds(mol):
    '''
    Helper function for divide_natural
    Finds groups of connected rotatable bonds
    If they are longer than 6 bonds, splits as evenly as possible
    '''
    rbs = [i[1:3] for i in find_rotatable_bonds(mol)]
    rb_id_groups = []
    while rbs:
        q = [rbs.pop()]
        new_group = []
        while q:
            working = q.pop()
            rems = [j for j in rbs if working[0] in j or working [1] in j]
            q.extend(rems)
            for i in rems:
                rbs.remove(i)
            new_group.append(working)
        # If the new group is too big, try to split it up
        if len(new_group) > 5:
            tmp = []
            for each_split in new_group:
                g1 = []
                g2 = []
                for each_bond in new_group:
                    if each_bond == each_split:
                        pass
                    elif sum([mol.distances[i][each_split[0]] for i in each_bond]) < sum([mol.distances[i][each_split[1]] for i in each_bond]):
                        g1.append(each_bond)
                    else:
                        g2.append(each_bond)
                tmp.append((abs(len(g1) - len(g2)), g1, g2))
            tmp.sort()
            rb_id_groups.append(tmp[0][1])
            rb_id_groups.append(tmp[0][2])
        else:
            rb_id_groups.append(new_group)
    return rb_id_groups

def attach_rigid_linkers(rotatable_bonds, mol):
    rotatable_sets = [set(chain(*g)) for g in rotatable_bonds]
    group_ids = []
    for each_idx, each_rotatable_set in enumerate(rotatable_sets):
        # Don't alter the original set - we'll want to reference it later
        each_tmp = [i for i in each_rotatable_set]
        # Walk through the set that we can alter, finding all neighbouring atoms not part of another rotatable bond
        working_idx = 0
        while working_idx < len(each_tmp):
            curr_atom = mol.atoms[each_tmp[working_idx]]
            neighbour_ids = [i for i in curr_atom.get_bond_ids() if i not in each_tmp]
            each_tmp.extend([i for i in neighbour_ids if not test_rotatable(mol.atoms[i], curr_atom)])
            working_idx += 1
        group_ids.append(each_tmp)
    return group_ids

def recombine_fragments(fragment_id_grps, mol, num_bonds_in_frag):
    combined_groups = []
    merged = []
    for base_id, each_base_group in enumerate(fragment_id_grps):
        done = False
        if base_id not in merged:
            each_base_set = set(each_base_group)
            for other_id, each_other_set in enumerate(fragment_id_grps):
                if each_base_group != each_other_set:
                    each_intersection = each_base_set.intersection(each_other_set)
                    if each_intersection:
                        base_difference = each_base_set.difference(each_intersection)
                        other_difference = set(each_other_set).difference(each_intersection)
                        # Get between-set bonds
                        link_unique_bonds_1 = []
                        for i, j in product(each_intersection, base_difference):
                            if mol.get_bond_order(i, j):
                                link_unique_bonds_1.append((j, i))
                        link_unique_bonds_2 = []
                        for i, j in product(each_intersection, other_difference):
                            if mol.get_bond_order(i, j):
                                link_unique_bonds_2.append((j, i))
                        # Iterate over pairs of bonds
                        for each_lu_bond_1, each_lu_bond_2 in product(link_unique_bonds_1, link_unique_bonds_2):
                            union_atom_id = 1
                            # If union atoms are connected to each other and the whole thing is cis, merge fragments
                            if mol.get_bond_order(each_lu_bond_1[union_atom_id], each_lu_bond_2[union_atom_id]):
                                if cos(mol.get_torsion(*(each_lu_bond_1 + each_lu_bond_2[::-1]))) > 0:
                                    if num_bonds_in_frag[base_id] + num_bonds_in_frag[other_id] <= 5:
                                        combined_groups.append(list(set(each_base_group + each_other_set)))
                                        merged.append(other_id)
                                        done = True
                                        break
            # If not merged with anything, leave unmerged
            else:
                if not done:
                    combined_groups.append(list(each_base_group))
    return [list(i) for i in combined_groups]

def divide_natural(mol, split):
    rot_bond_id_grps = group_rotatable_bonds(mol)
    group_sizes = [len(i) for i in rot_bond_id_grps]
    fragment_id_grps = attach_rigid_linkers(rot_bond_id_grps, mol)
    recombined_id_grps = recombine_fragments(fragment_id_grps, mol, group_sizes)
    return recombined_id_grps

#Return the return lists
def recombine(base_mol, group_conformers, subset_rotatable, subset_to_mol):
    count = 1
    for each_set in product(*group_conformers):
        count += 1
        rbs = []
        nvs = []
        # Read off the rotatable bonds from each element
        for each_mol, each_bonds, each_trans in zip(each_set, subset_rotatable, subset_to_mol):
            to_make = {tuple([each_trans[j] for j in i]):each_mol.get_torsion(*i) for i in each_bonds}
            rbs += to_make.keys()
            nvs += to_make.values()
        # combine them into a base conformer
        recombined_mol = make_rotamer(base_mol, rbs, nvs)
        yield recombined_mol

def divide_and_conquer(mol, final_detla, scaling, allow_inversion, split = [], vary_rings = 1):
    if not split:
        group_ids = divide_natural(mol, split)
    else:
        group_ids = divide_linear(mol, split)
    group_mols = [mol.copy_subset(i) for i in group_ids]
    mol_to_subset = [{q: i.index(q) for q in range(len(mol.atoms)) if q in i} for i in group_ids]
    subset_to_mol = [{v:k for k, v in i.items()} for i in mol_to_subset]
    # Cap broken bonds with dummy atoms to guarantee that rotatable bonds stay rotatable
    for each_group, each_mts, each_stm in zip(group_mols, mol_to_subset, subset_to_mol):
        tmp = [i for i in each_group.atoms]
        for each_subgroup_atom in tmp:
            each_mol_atom = mol.atoms[each_stm[each_subgroup_atom.get_id()]]
            if len(each_mol_atom.get_bond_ids()) > len(each_subgroup_atom.get_bond_ids()):
                missing_neighbour = [ i for i in each_mol_atom.get_bond_ids() if i not in each_mts]
                for each_missing_id in missing_neighbour:
                    each_group.add_atom(atom.atom(0, *mol.atoms[each_missing_id].coords))
                    each_group.add_bond(each_subgroup_atom.get_id(), len(each_group.atoms) - 1, 1)
                    each_mts[each_missing_id] = len(each_group.atoms) - 1
                    each_stm[len(each_group.atoms) - 1] = each_missing_id
    # Make a list of bonds that will be rotated around in the subgroups
    # Also mark bonds in subgroups that are in rings in the base mol as unrotatable
    mol_fix = []
    sub_fix = []
    sub_rotated = []
    for each_submol, each_sub_to_mol in zip(group_mols, subset_to_mol):
        each_sub_fix = []
        each_sub_rotated = []
        for each_rotatable_bond in find_rotatable_bonds(each_submol):
            full_mol_bond = [each_sub_to_mol[i] for i in each_rotatable_bond]
            if mol.is_in_ring(full_mol_bond[1], full_mol_bond[2]):
                each_sub_fix.append(each_rotatable_bond[1:3])
            else:
                mol_fix.append(full_mol_bond[1:3])
                each_sub_rotated.append(each_rotatable_bond)
        sub_fix.append(each_sub_fix)
        sub_rotated.append(each_sub_rotated)
    # Make the conformer ensembles for each section of the molecule
    group_conformers = []
    for each_subset, each_fix in zip(group_mols, sub_fix):
        base_ensemble = [q for q in make_all_rotations(each_subset, final_detla, scaling, allow_inversion, each_fix, False)]
        group_conformers.append(base_ensemble)
        allow_inversion = False # Only the first subgroup should have its first bond restricted
    # Recombine
    for recombined_mol in recombine(mol, group_conformers, sub_rotated, subset_to_mol):
        count = 0
        for each_conformer in make_all_rotations(recombined_mol, final_detla, scaling, allow_inversion, mol_fix, vary_rings):
            count += 1
            yield each_conformer


if __name__ == '__main__':
    import argparse
    import os.path
    description = '''
    UCONGA: Universal CONformer Generation and Analysis
    ---------------------------------------------------
    A conformer generation method that doesn't assume you're trying
    to dock a drug-like molecule to a protein.
    ---------------------------------------------------------------
    Nathaniel Gunby, Dr. Deborah Crittenden, Dr. Sarah Masters
    Department of Chemistry
    University of Canterbury
    Christchurch
    New Zealand
    '''
    scaling_help = 'Scaling factor for van der Waals radii, 0<s<1 (default=0.9)'
    delta_help = 'Angle in degrees to step the rotatable bonds by (default=30)'
    f_help = 'File format to output results in: cml, xyz, gms (GAMESS input geometry),'
    f_help += 'nw (nwchem input geometry), gauss (Gaussian input geometry)'
    o_help = 'Base name for the output files. Must contain a * which will be replaced '
    o_help += 'by the conformer number. Default = Conformer_*_${Input_name}'
    i_help = 'Ignore stereochemistry (treat enantiomeric conformers as identical).'
    r_help = 'Amount of ring conformers to generate (0=> hold rings fixed, 1 => reflect rings, 2=> full flip-of-fragments)'
    r_help += 'Default = 1 with divide-and-conquer and 2 without'
    a_help = 'Avoid using divide-and-conquer even when the molecule has more than 5 rotatable bonds (by default, the divide-and-conquer algoriothm is used for molecules with more than 5 rotatable bonds.)'
    b_help = 'Two atomic IDs (1-indexed) defining a bond to be broken for divide-and-conquer. This is ignored if -a is specified. Can be repeated. If unspecified, the molecule is broken at all multiple bonds and rings.'
    parser = argparse.ArgumentParser(description=description,
                            formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-s', '--scaling', help=scaling_help, type=float,
                        default=None)
    parser.add_argument('-d', '--delta', help=delta_help, type=int, default=30)
    parser.add_argument('-i', '--allow_inversion', help=i_help, action='store_true')
    parser.add_argument('-o', '--output_name', help=o_help)
    parser.add_argument('-f', '--output_format', help=f_help, default='cml')
    parser.add_argument('-r', '--ring_generation', help=r_help, default=-1, type=int)
    parser.add_argument('-a', '--avoid_division', help=a_help, action='store_true')
    parser.add_argument('-b', '--break_at', help=b_help, action='append', type=int, nargs=2)
    parser.add_argument('file_name', help='cml file containing the molecule')
    args = parser.parse_args()
    vary_rings = args.ring_generation
    fix_or_split = []
    try:
        mol = molecule.from_cml(args.file_name)
    except:
        msg = 'Reading molecule failed. Please check that the file a) exists and b) is a .cml molecule file'
        raise RuntimeError(msg)
    if args.output_name is None:
        in_name = os.path.normpath(args.file_name)
        in_path_name, in_file_name = os.path.split(in_name)
        base_file_name = os.path.join(in_path_name, 'Conformer_{0}_' + in_file_name)
        # Make sure that the file extension is appropriate to the contents of the file
        if args.output_format == 'xyz':
            base_file_name = os.path.splitext(base_file_name)[0] + '.xyz'
        elif args.output_format == 'gauss':
            base_file_name = os.path.splitext(base_file_name)[0] + '.gjf'
        elif args.output_format != 'cml':
            base_file_name= os.path.splitext(base_file_name)[0] + '.inp'
    else:
        base_file_name = args.output_name.replace('*', '{0}')
    if '{0}' not in base_file_name:
        raise(ValueError, "No wildcard in output file namer")
    if args.scaling is None:
      args.scaling = choose_scaling(mol)
    if args.scaling < 0 or args.scaling > 1:
        raise(ValueError, "van der Waals scaling factor not in 0<s<1)")
    if len(find_rotatable_bonds(mol)) < 5 or args.avoid_division:
        conf_gen_func = make_all_rotations
        if vary_rings == -1:
            vary_rings = 2
    else:
        conf_gen_func = divide_and_conquer
        if vary_rings == -1:
            vary_rings = 1
        if args.break_at:
            fix_or_split = [[i - 1 for i in pair]for pair in args.break_at]
    for idx, each_conformer in enumerate(conf_gen_func(mol, args.delta,
                                                            args.scaling,
                                                            args.allow_inversion,
                                                            fix_or_split,
                                                            vary_rings),
                                         1):
        with open(base_file_name.format(idx), 'w') as out_file:
            if args.output_format == 'cml':
                out_file.write(each_conformer.to_cml())
            else:
                out_file.write(each_conformer.to_xyz(args.output_format))
