import numpy
import math
from itertools import combinations_with_replacement, chain, groupby, compress
from collections import Counter
import molecule
import constants
import UCONGA_generate as UCONGA
matplotlib_available = False
cluster_available = False
try:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator
    matplotlib_available = True
except ImportError:
    pass
try:
    import scipy.cluster
    cluster_available = True
except ImportError:
    pass
import warnings
from os import path
# Molecule preparation functions


def canonicalise(mol):
    '''
    Canonicalise a molecule with respect to nuclear permutational symmetry
    For each symmetric group, choose the equivalent position that UCONGA_generate
        would generate
    '''
    Accepts: a molecule object
    rb = UCONGA.find_rotatable_bonds(mol)
    centralness = [sum(i) for i in mol.distances]
    classes = mol.get_morgan_equivalencies()
    # Normalise the coords for the reference w. r. t. the symmetry operations:
    for each_bond in rb:
        if centralness[each_bond[0]] > centralness[each_bond[-1]]:
            each_bond = each_bond[::-1]
        if UCONGA.is_symmetrical_rotor(each_bond, mol, classes):
            sym_degree = float(len(mol.atoms[each_bond[2]].search_away_from(each_bond[1])))
            increment = 2 * math.pi / sym_degree
            each_torsion = mol.get_torsion(*each_bond)
            while each_torsion > increment:
                each_torsion -= increment
            while each_torsion < 0:
                each_torsion += increment
            mol.set_torsion(*([i for i in each_bond] + [each_torsion]))
    # Group bonds by siblings - this is inefficient but not the limiting factor performance-wise
    new_dict = {}
    partial_symmetry_groups = []
    for each_id, each_bond in enumerate(rb):
        if centralness[each_bond[0]] > centralness[each_bond[-1]]:
            each_bond = each_bond[::-1]
        elder_bond_ids = []
        for i in UCONGA.find_older_sibling_ids(each_bond, each_id, mol, classes, new_dict):
            elder_bond_ids.append(int(i))
        app = [each_bond]
        for i in elder_bond_ids:
            i_bond = rb[i]
            if centralness[i_bond[0]] > centralness[i_bond[-1]]:
                i_bond = i_bond[::-1]
            app.append(i_bond)
        partial_symmetry_groups.append(app)
    partial_symmetry_groups.sort(key=len)
    # Remove the incomplete groups
    working_idx = 0
    while working_idx < len(partial_symmetry_groups) - 1:
        to_test = []
        for i in partial_symmetry_groups[working_idx + 1:]:
            if partial_symmetry_groups[working_idx][0] in i:
                del partial_symmetry_groups[working_idx]
                break
        else:
            working_idx += 1
    for each_whole_group in partial_symmetry_groups:
        ab = [int(math.degrees(mol.get_torsion(*i))) for i in each_whole_group]
        for idx, i in enumerate(ab):
            if i < 0:
                ab[idx] = i + 360
        ab.sort()
        for each_bond, each_torsion in zip(each_whole_group, ab):
            repacked = [i for i in each_bond] + [math.radians(each_torsion)]
            mol.set_torsion(*repacked)


def calculate_I_tensor(coords, weights):
    '''
    Calculate the inertia tensor of a molecule

    Accepts: a coordinate matrix
             a list of atomic weights

    Returns: the inertia tensor
    '''
    x = coords[:, 0]
    y = coords[:, 1]
    z = coords[:, 2]
    I_xx = numpy.sum(weights * (y * y + z * z))
    I_xy = numpy.sum(weights * x * y) * -1
    I_xz = numpy.sum(weights * x * z) * -1
    I_yy = numpy.sum(weights * (x * x + z * z))
    I_yz = numpy.sum(weights * y * z) * -1
    I_zz = numpy.sum(weights * (x * x + y * y))
    return numpy.array([[I_xx, I_xy, I_xz],
                        [I_xy, I_yy, I_yz],
                        [I_xz, I_yz, I_zz]])


def align_inertial(mol):
    '''
    Aligns a molecule so that its axes of inertia are aligned to the coordinate axes
    The primary axis (highest moment) will be aligned to the z-axis,
    while the tertiary (lowest moment) axis will be aligned to the x-axis

    Accepts: a molecule

    Returns: nothing, it modifies the molecule in-place
    '''
    c_weights = numpy.array([[constants.periodic_table[constants.periodic_list[i.num]]['mass']]
                             for i in mol.atoms])
    r_weights = c_weights.T[0]
    c_coords = mol.coord_matrix()
    # centre on the COM
    weighted_coords = c_weights * c_coords
    all_weight = sum(c_weights)
    com = weighted_coords.sum(axis=0) / all_weight
    c_coords -= com
    intertia_mat = calculate_I_tensor(c_coords, r_weights)
    moments, axes = numpy.linalg.eig(intertia_mat)
    tmp = [(j, k) for j, k in zip(moments, axes)]
    sorted_axes = numpy.array([i[1] for i in sorted(tmp)])
    c_new_coords = sorted_axes.T.dot(c_coords.T).T
    for a, c in zip(mol.atoms, c_new_coords):
        a.coords = c


def find_bbox(mol):
    """
    Find the bounding box dimensions along the inertial axes

    Accepts: a molecule

    Returns: The three bouding-box side lengths, from lowest to highest moment
             of inertia, as a numpy array
    """
    align_inertial(mol)
    # Find the eliptical radii along the axes of inertia
    x = [i.coords[0] for i in mol.atoms]
    r_x = max(x) - min(x)
    y = [i.coords[1] for i in mol.atoms]
    r_y = max(y) - min(y)
    z = [i.coords[2] for i in mol.atoms]
    r_z = max(z) - min(z)
    return numpy.array([r_x, r_y, r_z])


def prepare_angles(important_torsions, mols, allow_inversion):
    '''
    Read torsional data from a lost of molecules at once, in preparation for
    torsion-based clustering
    Accepts:
        A list of lists of atom ids for torsions of interest
        A list of molecules
        Whether or not enantiomers should be trated as identical
    Returns: A list of lists of torsion values for each molecule, in radians
    '''
    tc_angles = [[mol.get_torsion(*torsion) for torsion in important_torsions]
                 for mol in mols]
    if allow_inversion:
        new_angles = []
        for each_mol in tc_angles:
            if each_mol[0] < 0:
                new_angles.append([i * -1 for i in each_mol])
            else:
                new_angles.append(each_mol)
        tc_angles = new_angles
    return tc_angles
# Clustering-related functions


def ch_cluster(data, method):
    '''
    Perform clustering on an array of data and
    calculate the Calinski-Harabasz Criterion for various k
    See Calinski and Harabasz., Commun. Stat., 1974, p 1

    Accepts: an array of data, with individual data points as the rows
            a function which takes that data and a number of clusters,
            and returns a list of each data points' cluster ids, the cluster centers,
            the total distortion, and a list of the distance from each point to its nearest center

    Returns: a list of tuples containg: the criterion
                                        a list of each conformer's cluster id
                                        a list of the distance from each conformer to its cluster center
    Note that the list starts for two clusters, as the criterion cannot be calulated for one
    '''

    all_data = []
    num_points = len(data)
    std_devs = numpy.std(data, axis=0)
    # Remove any columns with a standard deviation of 0, which cannot be whitened
    data_to_use = numpy.array([i for i in compress(data.transpose(), std_devs)]).transpose()
    mean = numpy.mean(data_to_use, axis=0)
    if num_points > 15:
        num_points = int(2 * math.sqrt(num_points))
    for num_clusters in range(2, num_points):
        with warnings.catch_warnings(record=True) as warnings_log:
            mapping, codebook, total_distortion, point_distortions = method(data_to_use, num_clusters)
        if total_distortion == 0:
            warnings.warn('Clustering with %d clusters has too many degrees of freedom. Are some conformers degenerate?' % num_clusters, RuntimeWarning)
        else:
            counts = Counter(mapping)
            ssb = sum([counts[idx] * (numpy.linalg.norm(mean - i) ** 2)
                       for idx, i in enumerate(codebook)])
            ch_criterion = ((num_points - num_clusters) * ssb) / ((num_clusters - 1) * total_distortion)
            all_data.append([ch_criterion, mapping, point_distortions])
    return all_data


def cluster_kmeans(data, k):
    '''
    Performs k-means clustering on a numpy array of data.

    Accepts: an array of data, with individual data points as the rows
             The number of clusters to make

    Returns: a list of each data points' cluster ids
             a list of the cluster centers,
             the total distortion,
             a list of the distances from each point to its nearest center
    '''
    whitened_data = scipy.cluster.vq.whiten(data)
    codebook, total_distortion = scipy.cluster.vq.kmeans(whitened_data, k, iter=50)
    mapping, point_distortions = scipy.cluster.vq.vq(whitened_data, codebook)
    return mapping, codebook, total_distortion, point_distortions


def cluster_hierarchy(data, k):
    '''
    Performs hierarchical clustering on a numpy array of data

    Accepts: an array of data, with individual data points as the rows
             The number of clusters to make

    Returns: a list of each data points' cluster ids
             a list of the cluster centers,
             the total distortion,
             a list of the distances from each point to its nearest center
    '''
    distmat = scipy.spatial.distance.pdist(data)
    linkmat = scipy.cluster.hierarchy.linkage(distmat, method='average')
    mapping = scipy.cluster.hierarchy.fcluster(linkmat, k, criterion='maxclust')
    mapped_points = [(i, data[i]) for i in mapping]
    mapped_points.sort(key=lambda x: x[0])
    clusters = [[j[1] for j in i[1]] for i in groupby(mapped_points, key=lambda x: x[0])]
    codebook = numpy.array([numpy.array(i).mean(axis=0) for i in clusters])
    total_distortion = sum([numpy.linalg.norm(i[1] - codebook[i[0] - 1])
                            for i in mapped_points])
    return mapping, codebook, total_distortion, point_distortions


def choose_best_clustering(clustering):
    '''
    Choose the best cluster size according to the *first derivative* of the
    Calinski-Harabasz criterion
    Accecpts: a list of tuples of cluster details where the first item is the
                        Calinski-Harabasz metric, the second item is the codebook,
                        and the third item is the list of distances from points to
                        the closest cluster centers
    Returns:
            The codebook and list of distances for the best clustering
    '''
    ch_criteria = [i[0] for i in clustering]
    if len(ch_criteria) == 1:
        best_id = 0
    elif len(ch_criteria) == 2:
        best_id = ch_criteria.index(max(ch_criteria))
    else:
        choose_by = ([2 * (ch_criteria[0] - ch_criteria[1])] +
                     [2 * j - i - k for i, j, k in zip(ch_criteria[:-2], ch_criteria[1:-1], ch_criteria[2:])]
                     + [2 * (ch_criteria[-1] - ch_criteria[-2])])
        best_id = choose_by.index(max(choose_by))
    return clustering[best_id][1:]


def reorder(lst, codebook):
    '''
    Reorders the items in a list according to a codebook
    Accepts:
        A list of items to reorder
        A list of the indices they should be in
    Returns
        A new list
    '''
    tmp = [i for i in zip(codebook, lst)]
    tmp.sort()
    return [i[1] for i in tmp]
#### RMSD-related functions


def calc_rmsd(coords_1, coords_2):
    '''
    Calculates rmsd between two lists of coordinates
    No alignment is performed
    Accepts: Two sets of coordinates as numpy arrays
    Returns: The RMSD
    '''
    n = len(coords_1)
    assert n == len(coords_2)
    return math.sqrt(sum([numpy.linalg.norm(i - j) ** 2
                          for i, j in zip(coords_1, coords_2)]) / float(n))


def align(coords_1, coords_2, allow_inversion=True):
    '''
    Calculates a transformation matrix to minimise the rmsd between two sets of coordinates
    The rmsd is not calculated
    Accepts:
        Two sets of coordinates as numpy arrays
        Whether or not inversion is allowed in the transformation
    Returns:
        A transformation matrix as a numpy array

    The transformation is a combination of rotations, reflections, and inversion
    This uses the Schonemann soltion to the orthogonal procrustes problem if inversion is allowed
    see Schonemann, Psychometrika 1966, p.1
    If inversion is not allowed, the Kabsch algorithm is used.
    See Kabsch, Acta Crystallogr. Sect. A 1976, p. 922 and
    Kabsch, Acta Crystallogr. Sect. A 1978, p. 827.
    '''
    M = coords_1.transpose().dot(coords_2)
    U, s, V_H = numpy.linalg.svd(M)
    if allow_inversion:
        return U.dot(V_H)
    else:
        V = numpy.transpose(V_H)
        U_H = numpy.transpose(U)
        d = [1.0 for i in U_H]
        d[-1] = numpy.linalg.slogdet(V.dot(U_H))[0]
        D = numpy.diag(d)
        return U.dot(D.dot(V_H))


def calc_min_rmsd(coords_1, coords_2, allow_inversion=True):
    '''
    Aligns the second set of coordinates to the first and then calculates the rmsd
    Accepts:
        Two sets of coordinates as numpy arrays
        Whether or not inversion is allowed in the transformation
    Returns:
        The RMSD
    '''
    alignment = align(coords_1, coords_2, allow_inversion)
    aligned_1 = coords_1.dot(alignment)
    return calc_rmsd(aligned_1, coords_2)


def all_the_rmsds(molecules, allow_inversion=True):
    '''
    Calculates all the rmsds in a collection of molecules with pariwise alignment
    Accepts:
        A list of molecule objects
        Whether or not inversion is allowed during the alignment
    Returns:
        A matrix of all the pairwise RMSDS
    '''
    l = len(molecules)
    rmsds = [[0 for j in range(l)] for i in range(l)]
    for i in molecules:
        i.center()
    coords = [j.coord_matrix() for j in molecules]
    for i, j in combinations_with_replacement(range(l), 2):
        r = calc_min_rmsd(coords[i], coords[j], allow_inversion)
        rmsds[i][j] = r
        rmsds[j][i] = r
    return rmsds
###### Graphing functions


def bbox_scatter(points, labs, ax=None):
    '''
    Makes a scatter plot of the bounding boxes colored by clustering
    Accepts:
    A list of data points
    A list of the cluster ids of each data point
    An optional matplotlib axis
    '''

    ax = ax if ax is not None else plt.gca()
    ax.set_title('Scatter plot of bounding-box clustering', size='x-large')
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    colors = plt.cm.Set1(numpy.linspace(0, 1, max(labs) + 1 - min(labs)))
    percentiles = numpy.percentile(points, [75, 25], axis=0)
    iqrs = percentiles[0] - percentiles[1]
    labels = ('Tertiary axis of inertia ($\AA$)',
              'Secondary axis of inertia ($\AA$)',
              'Primary axis of inertia ($\AA$)')
    tmp = [(i, idx) for idx, i in enumerate(iqrs)]
    tmp.sort()
    x_idx = tmp[2][1]
    y_idx = tmp[1][1]
    ax.set_ylabel(labels[y_idx], size='x-large')
    ax.set_xlabel(labels[x_idx], size='x-large')
    c_array = [colors[i] for i in labs]
    ax.scatter(points[:, x_idx], points[:, y_idx], s=50, c=c_array, marker='+')
    plt.show()


def parallel_coordinates(data, torsion_labels, categories, allow_inversion, ax=None):
    '''
    Makes a parallel coordinates plot of the torsion angles colored by clustering
    Attempts to make all cluster centers as visually close to each other as possible,
    and all data points as visually close to their center as possible
    Accepts:
    A list of data points
    A list of the torsion labels as strings
    A list of the cluster ids of each data point
    Whether or not enantiomers should be treated as identical
    An optional matplotlib axis
    '''
    ax = ax if ax is not None else plt.gca()
    ax.set_title('Parallel coordinates plot of torsional clustering', size='x-large')
    colors = plt.cm.Set1(numpy.linspace(0, 1, max(categories) + 1 - min(categories)))
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_ylabel('Torsion angle value ($^\circ$)', size='x-large')
    ax.set_xlabel('Torsion label', size='x-large')
    label_positions = range(-360, 450, 90)
    ax.yaxis.set_major_locator(FixedLocator([math.radians(i) for i in label_positions]))
    ax.yaxis.set_ticklabels([str(i) for i in label_positions], size='large')
    x_coords = [0.3 * i for i in range(len(torsion_labels))]
    ax.xaxis.set_major_locator(FixedLocator(x_coords))
    ax.xaxis.set_ticklabels(torsion_labels, size='large')
    # The tricky thing about plotting angles on a flat graph is getting the wrapping right - is 0 or 360 more communicative?
    # To do this, find the centers of the clusters, wrap them to be as close to each other as possible,
    # and wrap the conformers in each cluster to be as close to its center as possible. That creates good wrapping
    for each_mol in data:
        if allow_inversion and each_mol[0] < 0:
            for idx, i in enumerate(each_mol):
                each_mol[idx] = i * -1
    all_centers = []
    clusters = [[j for j, k in zip(data, categories) if k == i]
                for i in range(max(categories) + 1)]
    for each_cluster in clusters:
        # To average angles, they need to be converted to sines and cosines:
        tmp = []
        for mol in each_cluster:
            for angle in mol:
                tmp.extend((math.sin(angle), math.cos(angle)))
        cart_conformers = numpy.array(tmp)
        average = numpy.mean(cart_conformers, axis=0)
        center = [math.atan2(s, c) for s, c in zip(average[::2], average[1::2])]
        all_centers.append(center)
    for each_center, each_cluster, each_color in zip(all_centers, clusters, colors):
        for idx, i in enumerate(each_center):
            j = all_centers[0][idx]

            if math.pi < j - i:
                each_center[idx] = i + 2 * math.pi
            elif math.pi < i - j:
                each_center[idx] = i - 2 * math.pi
        for each_conformers in each_cluster:
            for idx, i in enumerate(each_conformers):
                j = each_center[idx]
                if math.pi < j - i:
                    each_conformers[idx] = i + 2 * math.pi
                if math.pi < i - j:
                    each_conformers[idx] = i - 2 * math.pi
            ax.plot(x_coords, each_conformers, color=each_color)
    curr_ylim = [int(math.degrees(i)) for i in ax.get_ylim()]
    new_ylim = (90 * (curr_ylim[0] / 90) - 5, 5 + (90 * ((curr_ylim[1] + 89) / 90)))
    ax.set_ylim([math.radians(i) for i in new_ylim])
    plt.tight_layout()
    plt.show()


def greyscale_visualisation(matrix, max_weight=None, ax=None, filename=None):
    """
    Visulaise a matrix (all entries positive real) as a greyscale square diagram
    Light = small number, dark = large number
    Accepts:
        A matrix to visualise
        An optional value to set to black
        An optional matplotlib axis object
        An optional filename to save to instead of showing
    If a maximum weight (which will correspond to black) is not provided,
    2**ceiling(log2[max]) is used, where max is the largest entry in the matrix
    """
    ax = ax if ax is not None else plt.gca()
    ax.set_title('Greyscale visualisation of the RMSD matrix', size='x-large')
    if not max_weight:
        max_weight = 2 ** numpy.ceil(numpy.log(numpy.abs(matrix).max()) / numpy.log(2))
    i = ax.matshow(matrix, vmin=0, vmax=max_weight, cmap='Greys')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())
    ax.set_xlabel('Conformer Number', size='x-large')
    ax.set_ylabel('Conformer Number', size='x-large')
    ax.text(1.2, 0.5, 'RMSD similarity', size='x-large', transform=ax.transAxes,
            rotation='vertical', verticalalignment='center')
    plt.colorbar(i)
    if not filename:
        plt.show()
    else:
        plt.savefig(filename)

if __name__ == '__main__':
    import argparse
    import csv
    from sys import stdout
    from glob import glob
    description = '''
    UCONGA: Universal CONformer Generation and Analysis
    ---------------------------------------------------
    A conformer diversity analysis tool that doesn't assume you're trying
    to dock a drug-like molecule to a protein.
    ----------------------------------------------------------------------
    Nathaniel Gunby, Dr. Deborah Crittenden, Dr. Sarah Masters
    Department of Chemistry
    University of Canterbury
    Christchurch
    New Zealand
    '''
    kt_help = '''Number of clusters for torsion-based clustering.
        If omitted or negative, k will be chosen using the Calinski-Harabasz Criterion
        If 0, torsion-based clustering will not be performed'''
    kb_help = '''Number of clusters for bounding-box-based clustering.
                If omitted or negative, k will be chosen using the Calinski-Harabasz Criterion
                If 0, bounding-box-based clustering will not be performed'''
    a_help = 'Run automatically (do not make graphs)'
    n_help = 'Do not calculate the rmsd matrix (this gets expensive for large ensembles)'
    o_help = 'File to write the results to. Default: stdout'
    i_help = 'Treat enantiomeric conformers as identical? (Default = No)'
    u_help = 'RMSD cutoff in Angstroms to treat conformers as unique (default = 1)'
    c_help = 'Use hierachical clustering (default = k-means clustering)'
    r_help = ('How to arrange molecules for RMSD visualisation (t = torsion ' +
              'clustering, b = bounding-box clustering, i or unspecified = conformer id)')
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-t', '--k_torsional', help=kt_help, type=int, default=-1)
    parser.add_argument('-b', '--k_bounding', help=kb_help, type=int, default=-1)
    parser.add_argument('-o', '--output_name', help=o_help, default=stdout,
                        type=argparse.FileType('wb'))
    parser.add_argument('-r', '--rmsd_arrange', help=r_help, default='i')
    parser.add_argument('-n', '--no_rmsd', help=n_help, action='store_true')
    parser.add_argument('-a', '--automatic', help=a_help, action='store_true')
    parser.add_argument('-i', '--allow_inversion', help=i_help, action='store_true')
    parser.add_argument('-u', '--uniqueness_cutoff', type=float, help=u_help, default=1.0)
    parser.add_argument('-H', '--use_hierachical_clustering', help=c_help,
                        action='store_true')
    parser.add_argument('input_files', help='cml files containing the conformers' +
                                            ' to be analysed', nargs='+')
    # Prepare input
    args = parser.parse_args()
    input_names = args.input_files
    for idx, i in enumerate(input_names):
        if '*' in i:
        # If the shell is dumb (powershell/cmd.exe), glob input files ourselves
            input_names[idx:idx + 1] = glob(i)
    mols = [molecule.from_cml(i) for i in input_names]
    for i in mols:
        canonicalise(i)
    # Clean up the molecule names for pretty-printing
    mol_names = [path.splitext(path.split(i)[1])[0] for i in input_names]

    writer = csv.writer(args.output_name)
    if args.use_hierachical_clustering:
        cluster = cluster_hierarchy
    else:
        cluster = cluster_kmeans
    # Validate input
    if len(set([len(i.atoms) for i in mols])) != 1:
        raise ValueError('Not all molecules to be analysed have the same number ' +
                         'of atoms. Please retry with conformers of only one molecule')
    if args.rmsd_arrange not in ['t', 'b', 'i']:
        raise ValueError('RMSD arrangement must be by [t]orsion clustering, ' +
                         '[b]ounding box clustering, or conformer [i]d')
    if (not cluster_available) and (args.k_torsional or args.k_bounding):
        warnings.warn('Scipy not detected. Will not perform clustering')
    if (len(mol_names) <= 3) and (args.k_torsional or args.k_bounding):
        warnings.warn('Not enough conformers to perform clustering')
    if args.rmsd_arrange and args.no_rmsd:
        warnings.warn('Instructions on how to arrange rmsd will be ignored ' +
                      'because rmsds are not calculated', RuntimeWarning)
    if (args.rmsd_arrange == 't') and ((args.k_torsional == 0) or (not cluster_available)
                                       or (len(mol_names) <= 3)):
        raise ValueError('Cannot arrange conformers by torsion clustering if ' +
                         'torsion clustering is not performed')
    if args.rmsd_arrange == 'b' and args.k_bounding == 0:
        raise ValueError('Cannot arrange conformers by bounding-box clustering' +
                         'if bounding-box clustering is not performed')
    # Do the clustering
    clustering_results = [mol_names]
    clustering_headings = ['Conformer name']
    # Torsional clustering
    if args.k_torsional != 0 and len(mol_names) > 3 and cluster_available:
        backbone, b_t_m = mols[0].copy_without_H()
        important_torsions = [[b_t_m[at] for at in it] for it in backbone.all_torsions()]
        important_angles = prepare_angles(important_torsions, mols, args.allow_inversion)
        tc_angles = numpy.array([[i for i in chain(*[(math.sin(angle), math.cos(angle)) for angle in mol])]
                                 for mol in important_angles])
        if args.k_torsional > 0:
            writer.writerow(['Finding %d torsional clusters' % args.k_torsional])
            t_ordering, tmp1, tmp2, t_distances = cluster(tc_angles, args.k_torsional)
        else:
            writer.writerow(['Finding torsional clusters with Calinski-Harabasz criterion'])
            t_clustering = ch_cluster(tc_angles, cluster)
            # Output the quality of the clustering
            t_ordering, t_distances = choose_best_clustering(t_clustering)
        tc_headings = clustering_headings + ['Torsional cluster id',
                                             'Distance from torsional cluster center']
        tc_headings += ['-'.join([str(j + 1)for j in i]) for i in important_torsions]
        writer.writerow(tc_headings)
        tc_results = clustering_results + ([t_ordering, t_distances])
        for i in range(len(important_torsions)):
            tc_results.append([math.degrees(j[i]) for j in important_angles])
        writer.writerows([i for i in zip(*tc_results)])
    # Bonding-box clustering
    if args.k_bounding != 0 and len(mol_names) > 3 and cluster_available:
        tc_bbox = numpy.array([find_bbox(i) for i in mols])
        if args.k_bounding > 0:
            writer.writerow(['Finding %d bounding-box clusters' % args.k_bounding])
            b_ordering, tmp1, tmp2, b_distances = cluster(tc_bbox, args.k_bounding)
        else:
            writer.writerow(['Finding bounding-box clusters with Calinski-Harabasz Criterion'])
            b_clustering = ch_cluster(tc_bbox, cluster)
            # Output the quality of the clustering
            b_ordering, b_distances = choose_best_clustering(b_clustering)
        b_headings = clustering_headings + (['Bounding-box cluster id',
                                             'Distance from bounding-box cluster center',
                                             'Tertiary axis of inertia',
                                             'Secondary axis of inertia',
                                             'Primary axis of inertia'])
        b_results = clustering_results + [b_ordering, b_distances, tc_bbox[:, 0],
                                          tc_bbox[:, 1], tc_bbox[:, 2]]
        writer.writerow(b_headings)
        writer.writerows([i for i in zip(*b_results)])
    # Calculate the RMSD matrix and perform filtering if desired
    if not args.no_rmsd:
        writer.writerow(['RMSD matrix'])
        if args.rmsd_arrange == 't':
            reorder(mols, t_ordering)
        if args.rmsd_arrange == 'b':
            reorder(mols, b_ordering)
        rmsds = all_the_rmsds(mols, allow_inversion=args.allow_inversion)
        writer.writerow(['Conformers'] + mol_names)
        for i, j in zip(mol_names, rmsds):
            writer.writerow([i] + j)
        # Use the rmsds to list the unique conformers
        writer.writerow(['Unique conformers'])
        accepted_ids = []
        for idx, rmsd_list in enumerate(rmsds):
            maybe_unique = [j for jdx, j in enumerate(rmsd_list[:idx])
                            if jdx in accepted_ids]
            if len(maybe_unique) == 0 or min(maybe_unique) > args.uniqueness_cutoff:
                writer.writerow([mol_names[idx]])
                accepted_ids.append(idx)
    # Visualise things
    if matplotlib_available and not args.automatic:
        if not args.no_rmsd:
            greyscale_visualisation(rmsds)
        if matplotlib_available and args.k_torsional and len(mol_names) > 3:
            parallel_coordinates(important_angles, ['-'.join([str(j + 1)for j in i]) for i in important_torsions], t_ordering, args.allow_inversion)
        if matplotlib_available and args.k_bounding and len(mol_names) > 3:
            bbox_scatter(tc_bbox, b_ordering)
