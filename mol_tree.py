def find_all_indices(l, to_find):
    res = []
    start_idx = 0
    while True:
        try:
            res.append(l.index(to_find, start_idx))
            start_idx = res[-1] + 1
        except ValueError:
            return res


def find_lr(at_dict, descend_into, start_idx=-1):
    '''
        Find the 'lower right' (in the tree) atom with children to swap
        This is the innermost loop for automorphism-location purposes
        '''
    lower_right = at_dict[start_idx]
    descent_options = filter(lambda x: descend_into[x], lower_right[1])
    while descent_options:
        lower_right = at_dict[descent_options[-1]]
        descent_options = filter(lambda x: descend_into[x], lower_right[1])
    return lower_right


def flatten_tree(at_dict, start_idx):
    '''
    Convert the molecule from a tree representation to a list representation
    '''
    ret = [start_idx]
    for j in at_dict[start_idx][1]:
        ret.extend(flatten_tree(at_dict, j))
    return ret


def from_mol(mol):
    '''
    Represents a molecule as a tree of (atom_id, [child_ids], parent_id) tuples
    An atom's parent is its most central bonded neighbour
    If an atom has two more central bonded neighbours (e.g. the para carbon of toluene
    one is randomly chosen
    An atom's children are all less central bonded neighbours
    The parent of the most central atom(s) is atom -1
    For this reason, the tuples are stored in a dictionary rather than a list
    The parent of -1 is undefined, as it should never be accessed
    '''
    atom_dicts = {}
    total_distances = [sum(i) for i in mol.distances]
    parents = []
    most_central_atoms = []
    for each_atom in mol.atoms:
        neighbours = each_atom.get_bond_ids()
        distances = [total_distances[i] for i in neighbours]
        min_distances = min(distances)
        if min_distances >= total_distances[each_atom.get_id()]:
            parents.append(-1)
            most_central_atoms.append(each_atom.get_id())
        else:
            idx = distances.index(min_distances)
            parents.append(neighbours[idx])
            atom_dicts = {}
    for idx in [-1] + range(len(mol.atoms)):
        parent_idx = parents[idx]
        curr_children = find_all_indices(parents, idx)
        ret = [idx, curr_children, parent_idx]
        atom_dicts[idx] = ret
    return atom_dicts