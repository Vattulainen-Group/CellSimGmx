# pdb_writer.py
def write_pdb_file(atoms, filename):
    with open(filename, 'w') as f:
        for atom in atoms:
            f.write("ATOM  {atomID:<5d} {atom_name:<4s} {resname:<3s} {chainID:<1s}{resid:>4d}    {X:>8.3f}{Y:>8.3f}{Z:>8.3f}\n".format(**atom))


# test_pdb_writer.py
import os
from pdb_writer import write_pdb_file

def test_write_pdb_file():
    atoms = [
        {'atom_name': 'N', 'atomID': 1, 'resname': 'ALA', 'resid': 1, 'chainID': 'A', 'X': 1.0, 'Y': 2.0, 'Z': 3.0},
        {'atom_name': 'CA', 'atomID': 2, 'resname': 'ALA', 'resid': 1, 'chainID': 'A', 'X': 2.0, 'Y': 3.0, 'Z': 4.0},
    ]

    filename = 'test_output.pdb'
    write_pdb_file(atoms, filename)

    assert os.path.exists(filename)

    with open(filename, 'r') as f:
        lines = f.readlines()

    assert len(lines) == len(atoms)
    for line, atom in zip(lines, atoms):
        assert line.startswith('ATOM')
        assert atom['atom_name'] in line
        assert str(atom['atomID']) in line
        assert atom['resname'] in line
        assert atom['chainID'] in line
        assert str(atom['resid']) in line
        assert str(atom['X']) in line
        assert str(atom['Y']) in line
        assert str(atom['Z']) in line

    os.remove(filename)

if __name__ == "__main__":
    test_write_pdb_file()