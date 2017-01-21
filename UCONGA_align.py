import molecule
import UCONGA_analyse
from os import path
import numpy


def align_to(mol_to_align, ref_mol, allow_inversion, center_id):
    '''
    Aligns one molecule to another
    '''
    ref_coords = numpy.array([i.coords for i in ref_mol.atoms
                              if i.get_id() in center_id])
    to_align_center = numpy.array([i.coords for i in mol_to_align.atoms
                                   if i.get_id() in center_id])
    mol_to_align.translate(-1*to_align_center.mean(axis=0))
    coords_to_align = numpy.array([i.coords for i in mol_to_align.atoms
                                   if i.get_id() in center_id])
    aligner = UCONGA_analyse.align(coords_to_align, ref_coords, allow_inversion)
    aligned_coords = mol_to_align.coord_matrix().dot(aligner)
    new_mol = mol_to_align.copy()
    for each_atom, each_new_coords in zip(new_mol.atoms, aligned_coords):
        each_atom.coords = each_new_coords
    return new_mol


if __name__ == '__main__':
    import argparse
    description = '''
    UCONGA: Universal CONformer Generation and Analysis
    ----------------------------------------------------
    A tool to align different conformers as a utility for UCONGA_analyse
    ----------------------------------------------------
    Nathaniel Gunby, Dr. Deborah Crittenden, Dr. Sarah Masters
    Department of Chemistry
    University of Canterbury
    Christchurch
    New Zealand
    '''
    i_help = 'Treat enantiomeric conformers as identical? (Default = No)'
    f_help = ('File format to output results in: cml, xyz, gms (GAMESS input ' +
              'geometry), nw (nwchem input geometry), gauss (Gaussian input geometry)')
    c_help = 'Atom ids to use for alignment (default = all heavy atoms)'
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-i', '--allow_inversion', help=i_help, action='store_true')
    parser.add_argument('-f', '--output_format', help=f_help, default='cml')
    parser.add_argument('-c', '--center_id', help=c_help, action='append', type=int)
    parser.add_argument('ref_file', help='cml file containing the reference ' +
                        'conformer that all others will be aligned to')
    parser.add_argument('input_files', help='cml files containing the other ' +
                        'conformers to be algigned', nargs='+')
    args = parser.parse_args()
    input_names = args.input_files
    if len(input_names) == 1 and '*' in input_names[0]:
        # If the shell is dumb (powershell/cmd.exe), glob input files ourselves
        from glob import glob
        input_names = glob(input_names[0])
    mols = [molecule.from_cml(i) for i in input_names]
    ref_mol = molecule.from_cml(args.ref_file)
    ref_num_atoms = len(ref_mol.atoms)
    if args.center_id:
        center_ids = [i - 1 for i in args.center_id]
    else:
        center_ids = [i.get_id() for i in ref_mol.atoms if i.num > 1]
    for each_id in center_ids:
        if not (0 <= each_id < ref_num_atoms):
            raise ValueError('The atom id to align molecules at must be a valid ' +
                             'atom id for the molecule, i.e. an integer between ' +
                             'between 1 and %d inclusive.' % ref_num_atoms)
    for each_mol in mols:
        if len(each_mol.atoms) != ref_num_atoms:
            raise ValueError('Not all molecules to be aligned have the same ' +
                             'number of atoms. Please retry with conformers ' +
                             'of only one molecule')
    ref_center = numpy.array([i.coords for i in ref_mol.atoms
                              if i.get_id() in center_ids])
    ref_mol.translate(-1*ref_center.mean(axis=0))
    input_names = [path.split(i) for i in input_names]
    out_names = [path.join(i[0], 'Aligned_{0}.{1}'.format(i[1], args.output_format))
                 for i in input_names]
    for each_mol, each_name in zip(mols, out_names):
        aligned_mol = align_to(each_mol, ref_mol, args.allow_inversion, center_ids)
        with open(each_name, 'w') as each_file:
            if args.output_format == 'cml':
                each_file.write(aligned_mol.to_cml())
            else:
                each_file.write(aligned_mol.to_xyz(args.output_format))
    ref_dirname, ref_fname = path.split(args.ref_file)
    with open(path.join(ref_dirname, 'Aligned_{0}.{1}'.format(ref_fname, args.output_format)), 'w') as f:
        if args.output_format == 'cml':
            f.write(ref_mol.to_cml())
        else:
            f.write(ref_mol.to_xyz(args.output_format))
