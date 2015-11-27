from math import radians, degrees
import molecule
import ring_lib
import numpy
import bisect
import warnings


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
        '''
    for each_pair in mol.all_pairs():
        atom_pair = [mol.atoms[each] for each in each_pair]
        if not mol.is_in_ring(*each_pair) and test_pair(atom_pair, scaling) is False:
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
    heavy_neighbours = [[j for j in each.search_away_from(other.get_id()) if this_mol.atoms[j].num > 1]
                        for each, other in zip(ats, ats[::-1])]
    # Do both ends have heavy neighbours?
    if 0 in [len(i) for i in heavy_neighbours]:
        return False
    # Is either end nearly linear?
    elif True in [len(i) == 1 and abs(this_mol.get_angle(each.get_id(), i[0], other.get_id())) > radians(170)
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
    maxes = []
    bonds_by_B_atom = {}
    for idx, each_bond in enumerate(rotatable_bonds):
        max = []
        if centralness[each_bond[0]] > centralness[each_bond[-1]]:
            each_bond = each_bond[::-1]
        C_children = backbone.atoms[each_bond[2]].search_away_from(each_bond[1])
        if each_bond[1] in bonds_by_B_atom:
            sym_classes = [classes[i] for i in each_bond]
            related_bonds = filter(lambda x: [classes[i] for i in rotatable_bonds[x]] == sym_classes,
                                   bonds_by_B_atom[each_bond[1]])
            max.extend([str(i) for i in related_bonds])
            bonds_by_B_atom[each_bond[1]].append(idx)
        else:
            bonds_by_B_atom[each_bond[1]] = [idx]
        if len(C_children) > 1 and len(set([classes[i] for i in C_children])) == 1:
            C_child_torsions = [backbone.get_torsion(each_bond[0], each_bond[1], each_bond[2], i)
                                for i in C_children]
            C_child_torsions.sort()
            diffs = [int(degrees(j - i)) for i, j in zip(C_child_torsions[:-1], C_child_torsions[1:])]
            # Probably need some sort of rounding thing here
            if len(set(diffs)) == 1:
                max.append(360/len(C_children))
        if not max:
            max.append(360)
        if len(maxes) == 0 and allow_inversion:
            max = [i/2 for i in max if type(i) != str]
        maxes.append(max)
    return maxes

def make_all_rotations(mol, final_delta, scaling, allow_inversion):
    '''
    A generator that yields all valid rotamers of a molecule
    Scaling is a scaling factor as defined for test_pair and test_mol
    Delta is the amount (in degrees) to increment torsions by in-between trials
    '''
    backbone, backbone_to_mol = mol.copy_without_H()
    rotatable_bonds = find_rotatable_bonds(backbone)
    mol_rotatable_bonds = [[backbone_to_mol[i] for i in each]
                           for each in rotatable_bonds]
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
    for each_set_of_rings in ring_lib.all_ring_conformers(mol):
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
    parser = argparse.ArgumentParser(description=description,
                            formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-s', '--scaling', help=scaling_help, type=float,
                        default=0.9)
    parser.add_argument('-d', '--delta', help=delta_help, type=int, default=30)
    parser.add_argument('-i', '--allow_inversion', help=i_help, action='store_true')
    parser.add_argument('-o', '--output_name', help=o_help)
    parser.add_argument('-f', '--output_format', help=f_help, default='cml')
    parser.add_argument('file_name', help='cml file containing the molecule')
    args = parser.parse_args()
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
    if args.scaling < 0 or args.scaling > 1:
        raise(ValueError, "van der Waals scaling factor not in 0<s<1)")
    for idx, each_conformer in enumerate(make_all_rotations(mol, args.delta,
                                                            args.scaling,
                                                            args.allow_inversion),
                                         1):
        with open(base_file_name.format(idx), 'w') as out_file:
            if args.output_format == 'cml':
                out_file.write(each_conformer.to_cml())
            else:
                out_file.write(each_conformer.to_xyz(args.output_format))
