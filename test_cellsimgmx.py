# Description: This file contains the tests for the classes in the system.py file.
import os
import shutil
import glob

from settings_parser import CLIParser, JSONParser, ForcefieldParserGMX

from system import Particle
from system import Cell
from system import Layer
from system import System

from gromacs import Simulation
from gromacs import GromppWrapper
from gromacs import MDRunWrapper
from gromacs import ExecuteSimulations

import config

def test_particle_creation():
    p = Particle(1, 'A', 1.0, 1.0, [1.0, 1.0, 1.0])
    assert p.particle_id == 1
    assert p.get_type() == 'A'
    assert p.get_coordinates() == [1.0, 1.0, 1.0]

def test_changing_particle_type():
    p = Particle(1, 'A', 1.0, 1.0, [0.0, 0.0, 0.0])
    p.set_type('B', 2.0, 2.0)
    assert p.get_type() == 'B'

def test_particle_coordinates():
    p = Particle(1, 'A', 1.0, 1.0, [0.0, 0.0, 0.0])
    assert p.get_coordinates() == [0.0, 0.0, 0.0]
    p.set_coordinates([1.0, 1.0, 1.0])
    assert p.get_coordinates() == [1.0, 1.0, 1.0]

def test_particle_shift():
    p = Particle(1, 'A', 1.0, 1.0, [0.0, 0.0, 0.0])
    p.shift_x(1.0)
    p.shift_y(2.0)
    p.shift_z(3.0)
    assert p.get_coordinates() == [1.0, 2.0, 3.0]


def test_cell_creation():
    cell = Cell(values, 0, cli_arguments, ff_parser)
    assert cell.cell_id == 0
    assert cell.nnneighbours == {}

def test_create_surface_particles():
    cell = Cell(values, 0, cli_arguments, ff_parser)
    cell.create_surface_particles()
    assert cell.get_surface_particles() != []
    assert values['nr_of_particles'] == len(cell.get_surface_particles())
    assert cell.get_surface_particles()[0].get_type() == 'C'
    assert cell.get_surface_particles()[0].get_coordinates() == [0.0, 0.0, 0.0]

def test_shift_cell():
    cell = Cell(values, 0, cli_arguments, ff_parser)
    cell.create_surface_particles()
    cell.create_surface()
    
    initial_coordinate = cell.get_surface_particles()[0].get_coordinates()[0]
    cell.shift_cell(1.0, 2.0, 3.0)
    shifted_coordinate = cell.get_surface_particles()[0].get_coordinates()[0]
    assert shifted_coordinate == initial_coordinate + 1.0

def test_find_nearest_neighbours():
    cell = Cell(values, 0, cli_arguments, ff_parser)
    cell.create_surface_particles()
    cell.create_surface()
    cell.find_nearest_neighbours()
    for k, v in cell.nnneighbours.items():
        assert v != []

def test_create_monolayer():
    layer = Layer(0, values, cli_arguments, ff_parser)
    layer.create_monolayer()
    assert len(layer.get_cells()) == values['number_of_cells']

def test_create_system():
    system = System(values, cli_arguments, ff_parser)
    system.create_system()
    total_particles = values['nr_of_particles'] * values['number_of_cells'] * values['nr_of_layers']
    print(f'total particles asked in input: {total_particles}')
    print(f'actual nr of particles: {len(system.particles)}')
    assert len(system.particles) == total_particles
    assert len(system.cells) == values['number_of_cells']* values['nr_of_layers']

def test_fit_box_around_coords():
    pass

def test_build_gro_file_system():
    system = System(values, cli_arguments, ff_parser)
    system.create_system()
    system.build_gro_file_system()
    assert len(system.particles) == values['nr_of_particles'] * values['number_of_cells']* values['nr_of_layers'], "Mismatch in particle counts"
    assert len(system.cells) == values['number_of_cells']* values['nr_of_layers'], "Mismatch in particle counts"
    assert len(system.centered_system_coords) == values['nr_of_particles'] * values['number_of_cells']* values['nr_of_layers'], "Mismatch in particle counts"

def test_create_topology():
    system = System(values, cli_arguments, ff_parser)
    system.create_system()
    system.build_gro_file_system()
    system.create_topology()
    total_particles = values['nr_of_particles'] * values['number_of_cells']* values['nr_of_layers']
    coords = [particle.get_coordinates() for particle in system.particles]
    atomnames = [particle.get_type() for particle in system.particles]
    cell_ids = [cell.cell_id for cell in system.cells for particle in cell.get_surface_particles()]
    assert len(coords) == total_particles, "Mismatch in particle counts"
    assert len(atomnames) == total_particles, "Mismatch in particle counts"
    assert len(cell_ids) == total_particles, "Mismatch in particle counts"
    assert len(system.particles) == total_particles, "Mismatch in particle counts"


def test_grompp():
    # test if .gro and .top files are created correctly

    system = System(values, cli_arguments, ff_parser)
    system.create_system()
    system.build_gro_file_system()
    system.create_topology()
    system.construct_system_topology()

    sim = Simulation(cli_arguments, ff_parser, values)
    sim.write_mdp_minim()
    
    gro_file = system.groname
    output_dir = cli_arguments['output_dir']
    if not os.path.exists(f'{output_dir}test_gmx'):
        os.mkdir('test_gmx')
    else:
        shutil.rmtree('test_gmx')
        os.mkdir('test_gmx')
    grompp_path = f'{output_dir}test_gmx'
    shutil.move(f'{output_dir}{gro_file}', f'{output_dir}test_gmx')
    shutil.move('system.top', 'test_gmx')
    shutil.move('mdps/em.mdp', 'test_gmx')
    
    grompp = f'gmx grompp -p {grompp_path}/system.top -f {grompp_path}/em.mdp -c {grompp_path}/{gro_file}'
    os.system(grompp)

def test_write_mdp_minim():
    sim = Simulation(cli_arguments, ff_parser, values)
    sim.write_mdp_minim()
    path = os.path.join(cli_arguments['output_dir'], 'mdps', 'em.mdp')
    assert os.path.exists(path)

def test_write_mdp_eq_prod():
    sim = Simulation(cli_arguments, ff_parser, values)
    sim.write_mdp_eq_prod()
    ensemble = values['ensemble']
    path = os.path.join(cli_arguments['output_dir'], 'mdps', 'NVE_eq.mdp')
    assert os.path.exists(path)

    if ensemble == 'NVE':
        path_nve = os.path.join(cli_arguments['output_dir'], 'mdps', 'production.mdp')
        assert os.path.exists(path_nve)
    elif ensemble == 'NVT':
        path_nvt = os.path.join(cli_arguments['output_dir'], 'mdps', 'NVT_eq.mdp')
        assert os.path.exists(path_nvt)
        path_nvt_prod = os.path.join(cli_arguments['output_dir'], 'mdps', 'production.mdp')
        assert os.path.exists(path_nvt_prod)
    elif ensemble == 'NPT':
        path_npt = os.path.join(cli_arguments['output_dir'], 'mdps', 'NpT_eq.mdp')
        path_npt_prod = os.path.join(cli_arguments['output_dir'], 'mdps', 'production.mdp')
        path_nve = os.path.join(cli_arguments['output_dir'], 'mdps', 'NVT_eq.mdp')
        assert os.path.exists(path_nve)
        assert os.path.exists(path_npt)
        assert os.path.exists(path_npt_prod)

def test_GromppWrapper():
    # build a system and runs grompp on energy minimization
    system = System(values, cli_arguments, ff_parser)
    system.create_system()
    system.build_gro_file_system()
    system.create_topology()
    system.construct_system_topology()

    sim = Simulation(cli_arguments, ff_parser, values)
    sim.write_mdp_minim()

    def create_directory(sim_path):
            if not os.path.exists(sim_path):
                os.makedirs(sim_path)

    init_coord = glob.glob(f'SYSTEM*.gro')[0] if glob.glob(f'SYSTEM*.gro') else None
    topol = "system.top" #this is hardcoded anyway
    create_directory("em")
    GromppWrapper(topol, init_coord, "mdps/em.mdp", "em/em.tpr")

    assert os.path.exists("em/em.tpr")

def test_MDrunWrapper():
    # run energy minimization
    MDRunWrapper("em/em")

def test_ExecuteSimulations():
    system = System(values, cli_arguments, ff_parser)
    system.create_system()
    system.build_gro_file_system()
    system.create_topology()
    system.construct_system_topology()

    sim = Simulation(cli_arguments, ff_parser, values)
    sim.write_mdp_minim()
    sim.write_mdp_eq_prod()
    execute_sims = ExecuteSimulations()

def test_write_pdb_file():
    # assume centering always true
    system = System(values, cli_arguments, ff_parser)
    system.create_system()
    system.write_pdb_file()
    filename = system.pdb_filename

    path = os.path.join(cli_arguments['output_dir'], filename)
    assert os.path.exists(path)

    with open(filename, 'r') as f:
        lines = f.readlines()

    assert len(lines) == len(system.centered_system_coords), "Mismatch in number of lines and system particles."
    assert len(lines) == len(system.particles)

    for line, atom in zip(lines, system.centered_system_coords.values()):
        assert line.startswith('ATOM')
        assert atom['name'] in line
        assert str(atom['resid']) in line
        assert str(atom['chain_id']) in line
        coords = atom['coords']
        for coord in coords:
            assert f"{coord:>8.3f}" in line, f"Coordinate {coord} not found in line: {line}"

    os.remove(filename)



if __name__ == '__main__':
    cli_parser = CLIParser()
    cli_arguments = cli_parser.parse_args()

    json_parser = JSONParser(cli_arguments)
    json_parser.load_json()
    values = json_parser.extract_json_values()

    ff_parser = ForcefieldParserGMX(cli_arguments)
    ff_parser.parse_GMX_ff()
    
    cell = Cell(values, config.cell_id, cli_arguments, ff_parser)

    test_particle_creation()
    test_changing_particle_type()
    test_particle_coordinates()
    test_particle_shift()

    test_cell_creation()
    test_create_surface_particles()
    test_shift_cell()
    test_find_nearest_neighbours()
    test_create_monolayer()
    test_create_system()
    test_build_gro_file_system()
    test_create_topology()
    #test_grompp()
    # test_write_mdp_minim()
    # test_write_mdp_eq_prod()
    # test_GromppWrapper()
    # test_MDrunWrapper()
    # test_ExecuteSimulations()
    test_write_pdb_file()

