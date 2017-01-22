import linalg
import numpy
from itertools import chain
import warnings
from scipy.optimize import leastsq
from math import degrees


def find_plane_of_ring(centered_ring_coords):
    '''
    Returns the normal to the plane of best fit of a ring system
    Accepts:
        A numpy array of the coordinates of the atoms in the ring
    Returns:
        A numpy array of the vector normal to the plane of the ring
    '''
    p0 = [1.0, 1.0, 1.0, 1.0]
    def f_min(X, p):
        plane_xyz = p[0:3]
        distance = (plane_xyz*X).sum(axis=1) + p[3]
        return distance / numpy.linalg.norm(plane_xyz)
    def residuals(params, signal, X):
        return f_min(X, params)
    sol = leastsq(residuals, p0, args=(None,centered_ring_coords))[0]
    norm = numpy.array(sol[0:3])
    return norm


def is_flippable(mol, ring_at_ids):
    '''
    Test if a ring is flippable
    Rings which are noticeably non-planar are defined as being flippable
    Takes:
        A molecule object
        A list of atom ids of the ring
    Returns:
        A boolean
    '''
    # 3 points define a plane, and <4 atoms can't be non-planar
    if len(ring_at_ids) <= 3:
        return False
    # Center the ring
    ring_coords = numpy.array([mol.atoms[i].coords for i in ring_at_ids])
    center_of_ring = numpy.average(ring_coords, axis=0)
    ring_coords -= center_of_ring
    # Find the normal to the plane of the ring, and normalise it (set length to 1)
    with warnings.catch_warnings(record=True) as warnings_log:
        norm = linalg.normalise(find_plane_of_ring(ring_coords))
    if len([i for i in warnings_log if 'reached maxfev' in str(i.message)]):
        # leastsq in find_plane_of_rings didn't converge.
        # Likely to be caused by a flat gradient near the minimum
        # so it *shouldn't* affect anything
        # Let the user know just in case
        ring_name = ', '.join([str(i) for i in ring_at_ids])
        msg = 'Ring_lib.is_flippable() can\'t fit a plane to the ring with atoms %s. Continuing. This may result in identical conformers being produced, so please check the output ensemble'
        warnings.warn(msg %ring_name, RuntimeWarning)
    xy_align = linalg.rotation_from_axes(numpy.array([0.0, 0.0, 1.0]), norm)
    aligned_ring_coords = ring_coords.dot(xy_align)
    for each_at_coords in aligned_ring_coords:
        if each_at_coords[2] > 0.1: #Is this a suitable value? I don't know
            return True
    return False


def all_ring_conformers(mol, full_flip=2):
    '''
    Find all combinations of all ring conformers for all the rings in a molecule
    Accepts:
        A molecule object
        An integer decribing the degree of ring conformer generation to undertake:
            none (0), partial (1), or full (2)
    Returns:
        A list of nomlecule objects, each with a different combination of ring conformers
    '''
    all_ring_systems = find_ring_systems(mol)
    ring_conformer_combinations = [mol]
    for each_ring_system in all_ring_systems:
        if is_flippable(mol, each_ring_system):
            grouped_conformer_combinations = [find_ring_conformers(i, each_ring_system, full_flip)
                                              for i in ring_conformer_combinations]
            ring_conformer_combinations = [i for i in chain(*grouped_conformer_combinations)]
    return ring_conformer_combinations


def find_ring_conformers(base_mol, each_ring_system, full_flip=2):
    '''
    Finds all unique conformers for a given ring system
    Accepts:
        A molecule object
        A list of atom ids of the ring
        An integer decribing the degree of ring conformer generation to undertake:
            none (0), partial (1), or full (2)
    Returns:
        A list of molecule objects, each with a different combination of ring conformers
    '''
    if full_flip > 1:  # Full ring conformer generation
        ret = []
        found_conformers = []
        all_dihedrals = [i for i in base_mol.all_torsions()]
        idx = 1
        for each_base_conformer in base_ring_variations(base_mol, each_ring_system):
            torsions = [int(degrees(each_base_conformer.get_torsion(*i)))
                        for i in all_dihedrals]
            if ((torsions not in found_conformers) and
                 ([-1 * i for i in torsions] not in found_conformers)):
                ret.append(each_base_conformer)
                found_conformers.append(torsions)
                mirror_image = flip_ring(each_base_conformer, each_ring_system)
                torsions = [int(degrees(mirror_image.get_torsion(*i))) for i in all_dihedrals]
                if ((torsions not in found_conformers) and
                     ([-1 * i for i in torsions] not in found_conformers)):
                    ret.append(mirror_image)
                    found_conformers.append(torsions)
            idx += 1
    elif full_flip == 1:  # Partial ring conformer generation
        ret = [base_mol.copy(), flip_ring(base_mol, each_ring_system)]
    else:  # No ring conformer generation
        ret = [base_mol.copy()]
    return ret


def base_ring_variations(base_mol, each_ring_system):
    '''
    Finds all two-atom flips-of-fragments
    The flip of fragments is defined in Mekenyan et al, J. Chem. Inf. Model. 2005, p. 283
    Accepts:
        A molecule object
        A list of atom ids of the ring
    Returns:
        A list of molecule objects, each with a different combination of ring conformers
    '''
    aligned_mol = base_mol.copy()  # Don't overwrite the base molecule
    # Center the molecule on the ring to be flipped
    ring_coords = numpy.array([aligned_mol.atoms[i].coords for i in each_ring_system])
    center_of_ring = numpy.average(ring_coords, axis=0)
    aligned_mol.translate(-1*center_of_ring)
    # Align the ring to be flipped on the xy plane
    ring_coords = numpy.array([aligned_mol.atoms[i].coords for i in each_ring_system])
    norm = find_plane_of_ring(ring_coords)
    xy_align = linalg.rotation_from_axes(norm, numpy.array([0.0, 0.0, 1.0]))
    aligned_mol.rotate(xy_align)
    # Build the list of flips of fragments
    ret = [aligned_mol.copy()]
    ring_coords = numpy.array([aligned_mol.atoms[i].coords for i in each_ring_system])
    z_to_precision = [abs(int(round(100*i[2]))) for i in ring_coords]
    if len(set(z_to_precision)) > 1 and len(each_ring_system) > 4:
        # Make a list of two-atom groups in the ring
        tmp_neighbour_pairs = [[frozenset([i, j]) for j in aligned_mol.atoms[i].get_bond_ids()
                                if j in each_ring_system]
                               for i in each_ring_system]
        neighbour_pairs = set([i for i in chain(*tmp_neighbour_pairs)])
        for each_pair_to_flip in neighbour_pairs:
            # The pair needs a distinct order, which frozensets don't
            each_pair_to_flip = list(each_pair_to_flip)
            new_mol = aligned_mol.copy()
            junction
            for i, j in zip(each_pair_to_flip, each_pair_to_flip[::-1]):
                for k in new_mol.atoms[i].search_away_from(j):
                    if k in each_ring_system:
                        junction.append(k)
            # Don't flip the ring at a ring junction
            if len(junction) != 2:
                break
            # Don't flip atoms with pi-bonding
            if (base_mol.atoms[junction[0]].get_hybridisation() != 3 or
                 base_mol.atoms[junction[1]].get_hybridisation() != 3):
                break
            substituents = []
            for i in each_pair_to_flip + junction:
                for j in chain(*[new_mol.atoms[i].all_neighbours_away_from(*each_ring_system):
                    substituents.append(j)
            atoms_reference = []
            for i, j in zip(junction, each_pair_to_flip):
                for k in new_mol.atoms[i].search_away_from(j):
                    if k in each_ring_system:
                        atoms_reference.append(k)
            translate_by = sum([new_mol.atoms[i].coords for i in junction]) / -2.0
            new_mol.translate(translate_by)
            reference_point = sum([new_mol.atoms[i].coords for i in atoms_reference]) / 2.0
            reflect_by = linalg.reflection_plane(new_mol.atoms[junction[0]].coords, reference_point)
            for each_id in substituents + each_pair_to_flip:
                each_atom = new_mol.atoms[each_id]
                each_atom.coords = reflect_by.dot(each_atom.coords)
            for each_id in each_pair_to_flip + junction:
                flip_substituents(new_mol, each_ring_system, each_id)
            new_mol.translate(-1*translate_by)
            ret.append(new_mol)
    return ret


def flip_ring(base_mol, each_ring_system):
    '''
    Takes the mirror image of a ring skeleton while preserving stereocehmistry
    at all the individual ring atoms
    This includes flipping a chair or half-chair conformation
    Accepts:
        A molecule object
        A list of atom ids of the ring
    Returns:
        A molecule object
    '''
    mirror_image = base_mol.copy()
    z_mirror = numpy.array([1.0, 1.0, -1.0])
    for each_atom in mirror_image.atoms:
        each_atom.coords *= z_mirror
    # Fix the stereochemistry
    for each_id_ring_atom in each_ring_system:
        each_ring_atom = mirror_image.atoms[each_id_ring_atom]
        in_ring_neighbours = [mirror_image.atoms[i] for i in filter(lambda x: x in each_ring_system,
                                                                    each_ring_atom.get_bond_ids())]
        if len(in_ring_neighbours) <= 2:
            flip_substituents(mirror_image, each_ring_system, each_id_ring_atom)
    return mirror_image


def flip_substituents(mol, each_ring_system, at_id):
    '''
    Change the stereocehsmitry at atom <at_id> in molecule <mol>
    This is part of the flip-of-fragments operation of Mekenyan et al
    See Mekenyan et al, J. Chem. Inf. Model. 2005, p. 283
    Accepts:
        A molecule object
        A list of atom ids of the ring
        The atom id to be corrected
    Returns:
        Nothing, this is a destructive method
    '''
    ring_atom = mol.atoms[at_id]
    in_ring_neighbours = [mol.atoms[i] for i in ring_atom.get_bond_ids()
                          if i in each_ring_system]
    neighbour_ids = [i.get_id() for i in in_ring_neighbours]
    substituents = [mol.atoms[i] for i in ring_atom.all_neighbours_away_from(*neighbour_ids)]
    # Centre everything:
    translate_by = -1 * ring_atom.coords
    for i in in_ring_neighbours + substituents:
        i.translate(translate_by)
    second_reflection = linalg.reflection_plane(in_ring_neighbours[0].coords,
                                                in_ring_neighbours[1].coords)
    for each_atom in substituents:
        each_atom.coords = second_reflection.dot(each_atom.coords)
    translate_back = -1 * translate_by
    for each_atom in in_ring_neighbours + substituents:
        each_atom.translate(translate_back)


def find_ring_systems(mol):
    '''
    Groups all ring systems in a molecule that share atoms together
    Takes: a molecule molecule
    Returns: a list of lists of ids
    Each list in the return is all atoms that are connected by an unbroken set of rings
    That is, any two atoms are a) in the same ring
                               b) in fused rings
                               c) in bridged rings
    Spiro rings are not included because their conformations are not coupled
    (e.g a sprio[6,6] system will have the two rings flip independently
    where as in a cis-decalin, both must flip at the same time)
    '''
    ring_systems = {}
    new_key = 0
    rings = [i for i in mol.rings]
    while len(rings) > 0:
        curr_ring = [rings.pop()]
        for k, v in ring_systems.items():
            count = 0
            for i in v:
                if i in curr_ring[0]:
                    if count:
                        curr_ring.append(v)
                        del ring_systems[k]
                        break
                    else:
                        count += 1
        expanded_system = set([i for i in
                               chain(*curr_ring)])
        ring_systems[new_key] = expanded_system
        new_key += 1
    return ring_systems.values()
