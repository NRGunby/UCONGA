import numpy
import math
from itertools import combinations_with_replacement, chain, groupby, count, compress, cycle
from collections import Counter
import molecule
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

def find_best_fit(ref_coords, other_coords):
    '''
    Find the molecules in other_coords that is closest to ref_coords
    '''
    rmsds = [calc_min_rmsd(ref_coords, i) for i in other_coords]
    m = min(rmsds)
    return m, rmsds.index(m)

def best_fit_wrapper(all_ref_names, other_mols, other_mol_names, writer):
    '''
    Find the best fit (see find_best_fit) of multiple reference molecules
    and prints everything nicely to a csv file
    '''
    ref_mols = [molecule.from_cml(i) for i in all_ref_names]
    writer.writerow(['Matching supplied conformers to generated conformers'])
    writer.writerow(['Supplied conformer', 'RMSD (A)', 'Closest generation conformer'])
    for i in ref_mols + other_mols:
        i.center()
    all_ref_coords = [i.coord_matrix() for i in ref_mols]
    other_coords = [i.coord_matrix() for i in other_mols]
    for each_ref_mol, each_ref_name in zip(all_ref_coords, all_ref_names):
        ref_rmsd, ref_idx = find_best_fit(each_ref_mol, other_coords)
        writer.writerow([each_ref_name, ref_rmsd, other_mol_names[ref_idx]])

    writer.writerow([''])

def ch_cluster(data, method):
    '''
    Performs k-means clustering on a  array of data and
    calculate the Calinski-Harabasz Criterion for various k
    See Calinski and Harabasz., Commun. Stat., 1974, p 1
    Returns a list of (criterion, cluster) pairs, where each cluster is a list of datum ids:
    [[datum 1, datum 2,...], cluster2, cluster3,...]

    '''
    all_data = []
    num_points = len(data)
    std_devs = numpy.std(data, axis=0)
    # Remove any columns with a standard deviation of 0, which cannot be whitened
    data_to_use = numpy.array([i for i in compress(data.transpose(), std_devs)]).transpose()
    mean = numpy.mean(data_to_use, axis=0)
    if num_points > 15:
        num_points = int(2*math.sqrt(num_points))
    for num_clusters in range(2, num_points):
        with warnings.catch_warnings(record=True) as warnings_log:
            mapping, codebook, distortion = method(data_to_use, num_clusters, True)
        if distortion == 0:
            warnings.warn('Clustering with %d clusters has too many degrees of freedom. Are some conformers degenerate?' %num_clusters, RuntimeWarning)
        else:
            counts = Counter(mapping)
            ssb = sum([counts[idx]*(numpy.linalg.norm(mean - i)**2) for idx, i in enumerate(codebook)])
            ch_criterion = ((num_points - num_clusters)*ssb)/((num_clusters - 1) * distortion)
            all_data.append([ch_criterion, mapping, codebook])
    return all_data

def cluster_kmeans(data, k, dist=False):
    '''
    Performs k-means clustering on a numpy array of data.
    Returns a list of clusters, where each cluster is a list of ids
    '''
    whitened_data = scipy.cluster.vq.whiten(data)
    codebook, distortion = scipy.cluster.vq.kmeans(whitened_data, k, iter=50)
    mapping = scipy.cluster.vq.vq(whitened_data, codebook)[0]
    if not dist:
        return mapping, codebook
    else:
        return mapping, codebook, distortion

def cluster_hierarchy(data, k, dist=False):
    '''
    Performs hierarchical clustering on a numpy array of data
    '''
    distmat = scipy.spatial.distance.pdist(data)
    linkmat = scipy.cluster.hierarchy.linkage(distmat, method='average')
    mapping = scipy.cluster.hierarchy.fcluster(linkmat, k, criterion='maxclust')
    mapped_points = [(i, data[i]) for i in mapping]
    mapped_points.sort(key=lambda x: x[0])
    clusters = [[j[1] for j in i[1]] for i in groupby(mapped_points, key=lambda x: x[0])]
    codebook = numpy.array([numpy.array(i).mean(axis=0) for i in clusters])
    if not dist:
        return mapping, codebook
    else:
        distortion = sum([numpy.linalg.norm(i[1] - codebook[i[0] - 1]) for i in mapped_points])
        return mapping, codebook, distortion


def calc_rmsd(coords_1, coords_2):
    '''
    Calculates rmsd between two lists of coordinates
    No alignment is performed
    '''
    n = len(coords_1)
    assert n == len(coords_2)
    return math.sqrt(sum([numpy.linalg.norm(i - j)**2
                          for i, j in zip(coords_1, coords_2)])/float(n))

def align(coords_1, coords_2, allow_inversion=True):
    '''
    Calculates a transformation matrix to minimise the rmsd between two sets of coordinates
    The rmsd is not calculated
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
    '''
    alignment = align(coords_1, coords_2, allow_inversion)
    aligned_1 = coords_1.dot(alignment)
    return calc_rmsd(aligned_1, coords_2)

def all_the_rmsds(molecules, allow_inversion=True):
    '''
    Calculates all the rmsds in a collection of molecules
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

def parallel_coordinates_setup(torsion_labels, ax=None):
    '''
    Some code common to both the clustered and unclustered versions of parallel coordinates visulation
    In particular, setting up the plot area, axes labels, and x tick marks
    '''
    ax = ax if ax is not None else plt.gca()
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_ylabel('Angle ($^\circ$)', size='x-large')
    ax.set_xlabel('Torsion', size='x-large')
    label_positions = range(-360, 450, 90)
    ax.yaxis.set_major_locator(FixedLocator([math.radians(i) for i in label_positions]))
    ax.yaxis.set_ticklabels([str(i) for i in label_positions])
    x_coords = [0.3 * i for i in range(len(torsion_labels))]
    ax.xaxis.set_major_locator(FixedLocator(x_coords))
    ax.xaxis.set_ticklabels(torsion_labels, rotation=30, ha='right')
    return ax, x_coords

def parallel_coordinates_finish(ax, filename=None):
    '''
    Some code common to both the clustered and unclustered versions of parallel coordinates visulation
    In particular, adjustiing the axes limits and showing the graph
    '''
    curr_ylim = [int(math.degrees(i)) for i in ax.get_ylim()]
    new_ylim = (90*(curr_ylim[0]/90) - 5, 5 + (90*((curr_ylim[1] + 89)/90)))
    ax.set_ylim([math.radians(i) for i in new_ylim])
    plt.tight_layout()
    if not filename:
        plt.show()
    else:
        plt.savefig(filename)

def parallel_coordinates(clusters, cluster_centres, torsion_labels, ax=None, filename=None):
    '''
    Visualises a series of clusters using parallel coordinates
    Each coordinate is a dihedral angles
    See https://en.wikipedia.org/wiki/Parallel_coordinates for more on parallel coordinates
    Individual conformers are drawn with thinner and paler lines than the cluster centres
    '''
    ax, x_coords = parallel_coordinates_setup(torsion_labels, ax=None)
    print '######################################'
    print torsion_labels, len(torsion_labels)
    print x_coords, len(x_coords)
    print cluster_centres
    print [len(i) for i in cluster_centres]
    base_colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1),
                   (1, 1, 0), (0, 1, 1), (1, 0, 1),
                   (0.5, 0.5, 0), (0, 0.5, 0.5), (0.5, 0, 0.5)]
    for combo in zip(range(1, len(clusters) + 1), clusters, cluster_centres, cycle(base_colors)):
        each_id, each_cluster, each_center, each_color = combo
        ax.plot(x_coords, each_center, label='Cluster %d' %each_id, color=each_color, lw=5, zorder=5000)
        pale_color = tuple([i * 0.7 for i in each_color] + [0.6])
        for each_conformers in each_cluster:
            for idx, i in enumerate(each_conformers):
                j = each_center[idx]
                if math.pi < j - i:
                    each_conformers[idx] = i + 2*math.pi
                if math.pi < i - j:
                    each_conformers[idx] = i - 2*math.pi
            ax.plot(x_coords, each_conformers, color=pale_color)
#ax.legend().get_frame().set_alpha(0.5)
    parallel_coordinates_finish(ax, filename)

def nocluster_parallel_coordinates(conformers, torsion_labels, ax=None, filename=None):
    '''
    Visualises a series of unclustered conformers using parallel coordinates
    Each coordinate is a dihedral angles
    See https://en.wikipedia.org/wiki/Parallel_coordinates for more on parallel coordinates
    '''
    ax, x_coords = parallel_coordinates_setup(torsion_labels, ax=None)
    for each_conformer in conformers:
        ax.plot(x_coords, each_conformer, color='k')
    parallel_coordinates_finish(ax, filename)

def greyscale_visualisation(matrix, max_weight=None, ax=None, filename=None):
    """
    Visulaise a matrix (all entries positive real) as a greyscale square diagram
    Light = small, black = large
    If a maximum weight (which will correspond to black) is not provided,
    2**ceiling(log2[max]) is used, where max is the largest entry in the matrix
    """
    ax = ax if ax is not None else plt.gca()
    if not max_weight:
        max_weight = 2**numpy.ceil(numpy.log(numpy.abs(matrix).max())/numpy.log(2))
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

def plot_line(data, labels, ax=None, filename=None):
    '''
    Plot a line
    Data should be a series of (x, y) tuples
    '''
    x = [i[0] for i in data]
    y = [i[1] for i in data]
    ax = ax if ax is not None else plt.gca()
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xlabel(labels[0], size='x-large')
    ax.set_ylabel(labels[1], size='x-large')
    ax.plot(x, y, color='k')
    if not filename:
        plt.show()
    else:
        plt.savefig(filename)


# Some utility methods refactored out of __main__ to avoid excess clutter


def summarise_cluster(cluster_center, conformer_list, tc_angles):
    '''
    Generate summary statistics for a cluster
    '''
    ret = {'Average': [math.degrees(i) for i in cluster_center]}
    relevant_rad_torsions = numpy.array([tc_angles[i] for i in conformer_list])
    center_to_center = numpy.array([math.pi]) - numpy.mean(relevant_rad_torsions, 0)
    relevant_rad_torsions += center_to_center
    stdev = numpy.std(relevant_rad_torsions, 0)
    ret['Standard Deviation'] = [math.degrees(i) for i in stdev]
    ret['Range'] = [math.degrees(i) for i in numpy.ptp(relevant_rad_torsions, 0)]
    ret['Interquartile range'] = [math.degrees(i - j)
                                  for i, j in zip(numpy.percentile(relevant_rad_torsions, 75, 0),
                                                  numpy.percentile(relevant_rad_torsions, 25, 0))]
    return ret

def write_cluster_info(writer, ordering_to_print, rad_centers, important_torsions):
    '''
    Utility function to write a clustering and summary statistics to a csv file
    '''
    writer.writerow([' %i clusters found' %len(ordering_to_print)])
    for idx, conformer_list, cluster_center in zip(count(1), ordering_to_print, rad_centers):
        writer.writerow(['Cluster %i:' %idx])
        writer.writerow(['Conformers:'] + [mol_names[i] for i in conformer_list])
        # Now write some information about the clustering
        writer.writerow(['Dihedral:'] + ['-'.join([str(molecule.py_to_id(j)) for j in i]) for i in important_torsions])
        for label, values in summarise_cluster(cluster_center, conformer_list, tc_angles).items():
            writer.writerow([label] + values)
    writer.writerow([''])


def arrange_by_clustering(to_be_arranged, ordering):
    '''
    Rearrange a list according to a specified ordering
    '''
    tmp = [(ordering[idx], i) for idx, i in enumerate(to_be_arranged)]
    tmp.sort()
    return [i[1] for i in tmp]

def choose_best_clustering(ch_data, clustering):
    '''
    Choose the best cluster size according to the *first derivative* of the Calinski-Harabasz criterion
    '''
    ch_criteria = [i[1] for i in ch_data]
    if len(ch_data) == 1:
        best_id = 0
    elif len(ch_data) == 2:
        best_id = ch_data.index(max(ch_data))
    else:
        choose_by = ([2 * (ch_criteria[0] - ch_criteria[1])] +
                     [2*j - i - k for i, j, k in zip(ch_criteria[:-2], ch_criteria[1:-1], ch_criteria[2:])]
                     + [2 * (ch_criteria[-1] - ch_criteria[-2])])
        best_id = choose_by.index(max(choose_by))
    return clustering[best_id][1:]

def prepare_angles(important_torsions, mols, allow_inversion):
    '''
    A utility function for reading data
    '''
    tc_angles = [[mol.get_torsion(*torsion) for torsion in important_torsions] for mol in mols]
    if args.allow_inversion:
        new_angles = []
        for each_mol in tc_angles:
            if each_mol[0] < 0:
                new_angles.append([i*-1 for i in each_mol])
            else:
                new_angles.append(each_mol)
        tc_angles = new_angles
    return tc_angles

def show_ch_info(ch_labels, ch_data, writer):
    '''
    A utility function to format the details of choosing the number of clusters
    for output to a csv file
    '''
    writer.writerow(ch_labels)
    writer.writerows(ch_data)
    writer.writerow([''])
    if matplotlib_available:
        plot_line(ch_data, ch_labels)


if __name__ == '__main__':
    import argparse
    import csv
    from sys import stdout
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
    n_help = 'Do not perform clustering - calculate rmsd values only'
    k_help = '''Number of clusters for k-means clustering.
        If omitted, k will be chosen using the Calinski-Harabasz Criterion'''
    o_help = 'File to write the rmsd matrix to. Default: stdout'
    i_help = 'Treat enantiomeric conformers as identical? (Default = No)'
    u_help = 'RMSD cutoff in Angstroms to treat conformers as unique (default = 1)'
    c_help = 'Use hierachical clustering (default = k-means clustering)'
    r_help = 'Reference conformer(s) to find the best fit to (optional)'
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-n', '--nocluster', help=n_help, action='store_true')
    parser.add_argument('-k', '--k', help=k_help, type=int, default=0)
    parser.add_argument('-o', '--output_name', help=o_help, default=stdout,
                        type=argparse.FileType('wb'))
    parser.add_argument('-i', '--allow_inversion', help=i_help, action='store_true')
    parser.add_argument('-u', '--uniqueness_cutoff', type=float, help=u_help, default=1.0)
    parser.add_argument('-H', '--use_hierachical_clustering', help=c_help, action='store_true')
    parser.add_argument('-r', '--reference_names', help=r_help, action='append')
    parser.add_argument('input_files', help='cml files containing the conformers to be analysed',
                        nargs='+')
    # Prepare input
    args = parser.parse_args()
    input_names = args.input_files
    if len(input_names) == 1 and '*' in input_names[0]:
         # If the shell is dumb (powershell/cmd.exe), glob input files ourselves
        from glob import glob
        input_names = glob(input_names[0])
    mols = [molecule.from_cml(i) for i in input_names]
    mol_names = [path.splitext(path.split(i)[1])[0] for i in input_names]
    writer = csv.writer(args.output_name)
    backbone, b_t_m = mols[0].copy_without_H()
    important_torsions = [[b_t_m[at] for at in it] for it in backbone.all_torsions()]
    tc_angles = prepare_angles(important_torsions, mols, args.allow_inversion)
    to_cluster = numpy.array([[i for i in chain(*[(math.sin(angle), math.cos(angle)) for angle in mol])]
                    for mol in tc_angles])
    if args.use_hierachical_clustering:
        cluster = cluster_hierarchy
    else:
        cluster = cluster_kmeans
    # Validate input
    if len(set([len(i) for i in to_cluster])) != 1:
        raise ValueError, 'Not all molecules to be analysed have the same number of atoms. Please retry with conformers of only one molecule'
    if (not args.nocluster) and len(mol_names) > 3 and (not cluster_available):
        warnings.warn('scipy.cluster could not be imported. Continuing without clustering', RuntimeWarning)
    if (not args.nocluster) and len(mol_names) <= 3 :
        warnings.warn('More than 3 conformers are needed to cluster. Continuing without clustering', RuntimeWarning)
    if args.reference_names:
        best_fit_wrapper(args.reference_names, mols, mol_names, writer)
    if (not args.nocluster) and len(mol_names) > 3 and cluster_available:
        if args.k < 0:
            raise ValueError, 'Number of clusters (k) cannot be negative.'
        elif args.k == 0:
            clustering = ch_cluster(to_cluster, cluster)
            # Output the quality of the clustering
            ch_labels = ['k (number of clusters) ', 'Calinski-Harabasz Criterion']
            ch_data = [j for j in enumerate([i[0] for i in clustering], 2)]
            show_ch_info(ch_labels, ch_data, writer)
            ordering, sc_centers = choose_best_clustering(ch_data, clustering)
        else:
            ordering, sc_centers = cluster(to_cluster, args.k)
        rad_centers = [[math.atan2(s, c) for s,c in zip(each_center[::2], each_center[1::2])]for each_center in sc_centers]
        # Write the clustering to the output
        ordering_tmp = [i for i in enumerate(ordering)]
        ordering_tmp.sort(key = lambda x: x[1])
        ordering_to_print = [[j[0] for j in i[1]] for i in groupby(ordering_tmp, key = lambda x: x[1])]
        # Make a pretty picture of the clusters
        if matplotlib_available:
            confs = [[tc_angles[i] for i in j]for j in ordering_to_print]
            parallel_coordinates(confs, rad_centers, ['-'.join([str(molecule.py_to_id(j)) for j in i]) for i in important_torsions])
        write_cluster_info(writer, ordering_to_print, rad_centers, important_torsions)
        mol_names = arrange_by_clustering(mol_names, ordering)
        mols = arrange_by_clustering(mols, ordering)
    else: #Use different parallel coordinates
        nocluster_parallel_coordinates(tc_angles, ['-'.join([str(molecule.py_to_id(j)) for j in i]) for i in important_torsions])
    writer.writerow(['RMSD matrix'])
    rmsds = all_the_rmsds(mols, allow_inversion=args.allow_inversion)
    if matplotlib_available:
        greyscale_visualisation(rmsds)
    writer.writerow(['Conformers'] + mol_names)
    for i, j in zip(mol_names, rmsds):
        writer.writerow([i] + j)
    #Use the rmsds to list the unique conformers
    writer.writerow(['Unique conformers'])
    for idx, rmsd_list in enumerate(rmsds):
        maybe_unique = rmsd_list[:idx]
        if len(maybe_unique) == 0 or min(maybe_unique) > args.uniqueness_cutoff:
            writer.writerow([mol_names[idx]])