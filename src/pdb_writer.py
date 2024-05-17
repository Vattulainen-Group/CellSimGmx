# pdb_writer.py
def write_pdb_file(atoms, filename):
    with open(filename, 'w') as f:
        for atom in atoms:
            f.write("ATOM  {atomID:<5d} {atom_name:<4s} {resname:<3s} {chainID:<1s}{resid:>4d}    {X:>8.3f}{Y:>8.3f}{Z:>8.3f}\n".format(**atom))


# test_pdb_writer.py
import os
from pdb_writer import write_pdb_file

def test_write_pdb_file():
    atoms = []
    atomID = 1
    for x in range(3):
        for y in range(3):
            for z in range(3):
                atom = {'atom_name': 'CA', 'atomID': atomID, 'resname': 'ALA', 'resid': 1, 'chainID': 'A', 'X': float(x), 'Y': float(y), 'Z': float(z)}
                atoms.append(atom)
                atomID += 1

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
