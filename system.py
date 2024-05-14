import numpy as np
import matplotlib.pyplot as plt
import datetime
import logging
from scipy.spatial import KDTree    #for n-nearest neighbour search
import os
import shutil
import re

from settings_parser import CLIParser
from settings_parser import JSONParser
from settings_parser import ForcefieldParserGMX

import config

class Particle:
    ''' Class to represent a particle in a cell'''
    def __init__(self, particle_id: int, type: str, sigma: float, epsilon: float, coordinates=[0.0, 0.0, 0.0]):
        self.__particle_id = particle_id
        self.__type = type
        self.__sigma = sigma
        self.__epsilon = epsilon
        self.__coordinates = coordinates
        config.particle_id += 1

    @property
    def particle_id(self) -> int:
        return self.__particle_id
    
    def set_type(self, new_type: str, new_sigma: float, new_epsilon: float) -> None:
        ''' Takes in an instance of Particle and changes type, sigma and epsilon.'''
        self.__type = new_type
        self.__sigma = new_sigma
        self.__epsilon = new_epsilon

    def set_coordinates(self, new_coordinates: list) -> None:
        ''' Takes in an instance of Particle and changes coordinates.'''
        self.__coordinates = new_coordinates

    def shift_x(self, x_offset: float) -> None:
        ''' Takes in an instance of Particle and shifts the x-coordinate.'''
        self.__coordinates[0] += x_offset

    def shift_y(self, y_offset: float) -> None:
        ''' Takes in an instance of Particle and shifts the y-coordinate.'''
        self.__coordinates[1] += y_offset

    def shift_z(self, z_offset: float) -> None:
        ''' Takes in an instance of Particle and shifts the z-coordinate.'''
        self.__coordinates[2] += z_offset

    def get_coordinates(self) -> list:
        return self.__coordinates
    
    def get_type(self) -> str:
        return self.__type

    def __str__(self) -> str:
        return f'Particle ID: {self.__particle_id}, Type: {self.__type}, coordinates: {self.__coordinates}'


class Cell:
    ''' Class to represent a cell in a system.'''
    def __init__(self, json_values: dict, cell_id: int, cli_arguments: dict, ff_parser: ForcefieldParserGMX):
        self.__cli_arguments = cli_arguments
        self.__ff_parser = ff_parser
        self.__json_values = json_values
        self.__cell_id = cell_id

        self.__itpname = None
        self.__groname = None
        
        self.__surface_particles = []                                
        self.__cytosol_particles = []                                
        self.__nnneighbours = {}        # dict that stores the indices and atomnames of n-nearest neighbours in the membrane
        config.cell_id += 1             # increment the cell_id each time a new cell is created

    def __str__(self) -> str:
        return f'Cell ID: {self.__cell_id}, Number of particles: {len(self.__surface_particles)}'
    
    @property
    def cell_id(self) -> int:
        return self.__cell_id
    
    @property
    def nnneighbours(self) -> dict:
        return self.__nnneighbours

    def get_surface_particles(self) -> list:
        return self.__surface_particles
    
    def create_surface_particles(self) -> None:
        ''' Create particle instances on the surface of the cell'''
        nr_of_particles = self.__json_values["nr_of_particles"]

        # create particle objects, leave coordinates empty for now
        for i in range(nr_of_particles):
            if i == 0:
                self.__surface_particles.append(Particle(config.particle_id, 'C', 1.0, 1.0, [0.0, 0.0, 0.0]))
            else:
                self.__surface_particles.append(Particle(config.particle_id, 'M1', 1.0, 1.0, []))

    def create_cytosol_particles(self) -> None:
        pass

    def create_surface(self) -> 'list[Particle]':
        ''' Output: self.__surface_particles (list)'''
        
        nr_of_particles = self.__json_values["nr_of_particles"]
        cell_radius = self.__json_values["cell_radius"]
        shape = self.__json_values["initial_packing_shape"]

        if shape == "spherical":                            # only supported shape for now
            theta = np.pi * (3.0 - np.sqrt(5.0))            # the Golden angle in radians
            for i in range(1, nr_of_particles):
                y = 1 - (i / float(nr_of_particles)) * 2    # such that y-coordinates are between -1,1
                radius = np.sqrt(1 - y * y) * cell_radius   # then scale coords by requested radius (in nm)
                phi = i * theta
                x = np.cos(phi) * radius
                z = np.sin(phi) * radius
                coordinates = [x, y * cell_radius, z]
                self.__surface_particles[i].set_coordinates(coordinates)
                

        ### if --verbose flag enabled, display an image plot of the packed cell as a reference
        if self.__cli_arguments['verbose']:
            fig = plt.figure(figsize=(10, 8)) 
            ax = fig.add_subplot(111, projection='3d')

            x = [particle.get_coordinates()[0] for particle in self.__surface_particles]
            y = [particle.get_coordinates()[1] for particle in self.__surface_particles]
            z = [particle.get_coordinates()[2] for particle in self.__surface_particles]

            colors = ['r' if particle.get_type() == 'C' else 'b' for particle in self.__surface_particles]

            ax.scatter(x, y, z, c=colors, marker='o')
            ax.set_title('Packing of beads on cell surface')

            now = datetime.datetime.now()
            figname = "CELL_packing-{}.png".format(now.strftime("%H-%M-%S"))
            plt.savefig(f"{self.__cli_arguments['output_dir']}/{figname}")
        
        return self.__surface_particles
    
    def shift_cell(self, x_offset, y_offset, z_offset) -> None:
        ''' Shift the cell in the x, y, z directions, used for creating a layer of cells'''
        for particle in self.__surface_particles:
            particle.shift_x(x_offset)
            particle.shift_y(y_offset)
            particle.shift_z(z_offset)
        
    
    def randomise_surface(self) -> None:
        # vary the bead size
        pass
    
    def add_cytosol(self) -> None:
        pass

    def create_junctions(self) -> None:
        pass

    def find_nearest_neighbours(self) -> None:
        """
        Uses particle list from create_surface() to determine the n-nearest neighbours on the surface membrane, if enabled by the user in 'input.JSON'. 
            
        Output:
            A dict (self.nnneighbours) with as keys the indices of the connected neighbours and as value their atomnames. 
                
            If --verbose is enabled:
                A plot 'CELL_surface_bonds-{time}.png' of the n-neighbour scheme based on the user input to validate. 
        """
        nearest_neighbour_springs = int(self.__json_values["nearest_neighbour_springs"])
        
        if nearest_neighbour_springs != "off":
            logging.info(f"Nearest neighbour springs are set to '{nearest_neighbour_springs}' neighbours")
            
            coords = [particle.get_coordinates() for particle in self.__surface_particles]
            atomnames = [particle.get_type() for particle in self.__surface_particles]

            #construct the nearest neigbour information
            tree = KDTree(coords)
            self.__nnneighbours = {}

            for index, (coord, atomname) in enumerate(zip(coords, atomnames)):
                _, neighbour_indices = tree.query(coord, k=nearest_neighbour_springs+1) 
                key = f"{index + 1} {atomname}"
                #save the neighbour information as a tuple[int, str] where int = index and str = atomname
                self.__nnneighbours[key] = [(i+1, atomnames[i]) for i in neighbour_indices if i != index]
            
            if '1 C' in self.__nnneighbours: #remove the entry of the center bead from this dict!
                del self.__nnneighbours['1 C']
            
            if self.__cli_arguments['verbose']:
                print(f"\nVERBOSE MODE ENABLED. You set n-nearest neighbour springs to '{nearest_neighbour_springs}'.")
                
                x, y, z = zip(*coords) #extract the coordinates of the particles

                fig = plt.figure(figsize=(15, 8))

                # Create two plots, one with all surface bonds, and one with only a single atom and its neighbours
                # to verify for the user (a graphical check is easier than manually checking the topology)
                ax_all = fig.add_subplot(1, 2, 1, projection='3d')
                ax_all.scatter(x, y, z)
                for key, neighbours in self.__nnneighbours.items():
                    index, atom_name = key.split()
                    index = int(index)
                    for neighbour_index, neighbour_name in neighbours:
                        neighbour_index = int(neighbour_index)
                        ax_all.plot([x[index - 1], x[neighbour_index - 1]],
                                    [y[index - 1], y[neighbour_index - 1]],
                                    [z[index - 1], z[neighbour_index - 1]])

                ax_all.set_xlabel('X')
                ax_all.set_ylabel('Y')
                ax_all.set_zlabel('Z')
                ax_all.set_title('A. CELL: all surface bonds')

                for key in self.__nnneighbours.keys():
                    index, name = key.split()
                    if int(index) == 178:
                        selected_atom_name = name
                        break
                        # need both the index and atomname to find the right key
                
                #this atom (178) is relatively central, use it to draw surface bonds
                # can pick any other index here if needed
                atom_key = f"{178} {selected_atom_name}"
                neighbours = self.__nnneighbours.get(atom_key, [])

                ax_one_atom = fig.add_subplot(1, 2, 2, projection='3d')
                ax_one_atom.scatter(x, y, z)
                for neighbour_index, neighbour_name in neighbours:
                    neighbour_index = int(neighbour_index)
                    ax_one_atom.plot([x[178 - 1], x[neighbour_index - 1]],
                                    [y[178 - 1], y[neighbour_index - 1]],
                                    [z[178 - 1], z[neighbour_index - 1]])

                ax_one_atom.set_xlabel('X')
                ax_one_atom.set_ylabel('Y')
                ax_one_atom.set_zlabel('Z')
                ax_one_atom.set_title(f'B. CELL: single atom and {nearest_neighbour_springs} neighbours')

                now = datetime.datetime.now()
                figname = "CELL_surface_bonds-{}.png".format(now.strftime("%H-%M-%S"))
                plt.savefig(f"{self.__cli_arguments['output_dir']}/{figname}")
                print(f"Saved a figure of n-nearest neighbour springs to '{self.__cli_arguments['output_dir']}/{figname}'")
                logging.info(f"Saved a figure of n-nearest neighbour springs to '{self.__cli_arguments['output_dir']}/{figname}'")
            
        else:
            #if nearest neighbour springs are disabled then do nothing in this function but do tell the user
            logging.warning(f"Surface bonds (n-nearest neighbour) is set to '{nearest_neighbour_springs}'")
            pass

class Layer:
    ''' Class to represent a layer in a system.'''
    def __init__(self, layer_id: int, json_values: dict, cli_arguments: dict, ff_parser: ForcefieldParserGMX):
        # assume input gives number of layers and number of cells per layer
        self._layer_id = layer_id
        self._json_values = json_values
        self._cli_arguments = cli_arguments
        self._ff_parser = ff_parser

        self._itpname = None
        self._groname = None

        self._layer_particles = []
        self.__layer_cells = []
        self._nnneighbours = {}

        config.layer_id += 1

    def create_monolayer(self):
        nr_of_cells = self._json_values["number_of_cells"]
        nr_of_particles = self._json_values["nr_of_particles"]

        offset = 4
        # Determine the number of cells along each axis
        grid_size_x = int(np.ceil(np.sqrt(nr_of_cells)))
        grid_size_y = int(np.ceil(nr_of_cells / grid_size_x))
        grid_size_z = 1  # Limiting the grid size to 1 layer in the z-direction

        # Generate the new coordinates based on the offset in x and y
        for i in range(grid_size_x):
            for j in range(grid_size_y):
                for k in range(grid_size_z):
                    new_cell = Cell(self._json_values, config.cell_id, self._cli_arguments, self._ff_parser)
                    new_cell.create_surface_particles()
                    new_cell.create_surface()
                    new_cell.find_nearest_neighbours()
                    new_cell.shift_cell(i * offset, j * offset, k * offset)
                    for particle in new_cell.get_surface_particles():
                        self._layer_particles.append(particle)
                    self.__layer_cells.append(new_cell)


        if self._cli_arguments['verbose']:
            print(f"\nVERBOSE MODE ENABLED. You set the number of cells to '{nr_of_cells}' and the number of particles per cell to '{nr_of_particles}'.")
            fig = plt.figure() 
            ax = fig.add_subplot(111, projection='3d')

            for cell in self._layer_particles:
                x = [particle.get_coordinates()[0] for particle in cell.__surface_particles]
                y = [particle.get_coordinates()[1] for particle in cell.__surface_particles]
                z = [particle.get_coordinates()[2] for particle in cell.__surface_particles]

                colors = ['r' if particle.get_type() == 'C' else 'b' for particle in cell.__surface_particles]

                ax.scatter(x, y, z, c=colors, marker='o')
                ax.axes.set_xlim3d(left=0, right=20) 
                ax.axes.set_ylim3d(bottom=0, top=20) 
                ax.axes.set_zlim3d(bottom=-10, top=10) 
                ax.set_title('Packing of beads in layer')

            now = datetime.datetime.now()
            figname = "Layer_packing-{}.png".format(now.strftime("%H-%M-%S"))
            plt.savefig(f"{self._cli_arguments['output_dir']}/{figname}")
        
    def get_cells(self) -> list:
        return self.__layer_cells    
  

class System:
    def __init__(self, json_values: dict, cli_arguments: dict, ff_parser: ForcefieldParserGMX):
        self.__json_values = json_values
        self.__cli_arguments = cli_arguments
        self.__ff_parser = ff_parser
        self.__particles = []
        self.__cells = []
        self.centered_system_coords = {}

    @property
    def particles(self) -> list:
        return self.__particles
    
    @property
    def cells(self) -> list:
        return self.__cells

    def create_system(self):
        number_of_cells = self.__json_values["number_of_cells"]
        number_of_layers = self.__json_values["nr_of_layers"]
        number_of_particles = self.__json_values["nr_of_particles"]
        print(f'Creating a system with {number_of_cells} cells and {number_of_layers} layers')
        
        if number_of_layers == 1:
            layer = Layer(config.layer_id, self.__json_values, self.__cli_arguments, self.__ff_parser)
            layer.create_monolayer()

            for cell in layer.get_cells():
                self.__cells.append(cell)
                for particle in cell.get_surface_particles():
                    self.__particles.append(particle)

            print(f'Created a system with {len(self.__particles)} particles')
            print(f'Created a system with {len(self.__cells)} cells')

        if number_of_layers > 1:
            # not yet implemented
            pass

    def fit_box_around_coord(self):
        """
        Fits closest fitting (cubic) box around the given coordinates.
        
        Returns:
            box_size_x, box_size_y, box_size_y (floats) - Vectors of the box.
        """
        
        min_x = min_y = min_z = float("inf")
        max_x = max_y = max_z = float("-inf")

        coordinates = [particle.get_coordinates() for particle in self.__particles]
        for coord in coordinates:
            x, y, z = coord
            # Find minimum and maximum coordinates to figure out where box needs to go
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            min_z = min(min_z, z)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
            max_z = max(max_z, z)

        box_size_x = abs(max_x - min_x) #length of box vectors
        box_size_y = abs(max_y - min_y)
        box_size_z = abs(max_z - min_z)
        
        return box_size_x, box_size_y, box_size_z

    def build_gro_file_system(self):
        box_coord_offset = self.__json_values["box_coord_offset"] #used for box fitting
        nr_of_particles = self.__json_values["nr_of_particles"] + 1 #including center-particle
        resid = 0
        coords = [particle.get_coordinates() for particle in self.__particles]
        atomnames = [particle.get_type() for particle in self.__particles]
        
        centering = True # use to enable or disable centering in the box (useful for debugging)
        
        ## Centering of the system coordinates, add later:
        total_coords = len(coords) 
        
        #Sums up all coordinates, then divides over the total number of atoms/coordinates...
        sum_x = sum(float(coord[0]) for coord in coords)
        sum_y = sum(float(coord[1]) for coord in coords)
        sum_z = sum(float(coord[2]) for coord in coords)
        
        #... to find geometric center in each Cartesian direction
        center_x = sum_x / total_coords
        center_y = sum_y / total_coords
        center_z = sum_z / total_coords
        
        # #calculate the required box size (add offset later, in case of 0, nothing happens)
        box_size = self.fit_box_around_coord()
        
        extra_box_size = (
            box_size[0] + box_coord_offset,
            box_size[1] + box_coord_offset,
            box_size[2] + box_coord_offset
        )

        # Note: (0,0,0) is recognized as the box origin in GMX, thus need to translate the coordinates based
        # on the box dimensions. Calculate the translation factor of the positions to the geometric center      
        trans_x = -(center_x - (extra_box_size[0] / 2))
        trans_y = -(center_y - (extra_box_size[1] / 2))
        trans_z = -(center_z - (extra_box_size[2] / 2))
        
        # Move the atom coordinates based on the calculated translation factor
        for atom_index, (coord, atomname) in enumerate(zip(coords, atomnames)):
            name = atomname
            x, y, z = coord

            x_translated = x + trans_x
            y_translated = y + trans_y
            z_translated = z + trans_z
                
            self.centered_system_coords[atom_index] = {
                "name": name,
                "coords": [x_translated, y_translated, z_translated]
            }

        now = datetime.datetime.now()
        gro_header = f"GRO file of SYSTEM with {str(len(self.__particles))} particles written at {now.strftime('%H:%M:%S')}\n"

        self.groname = "SYSTEM-{}.gro".format(now.strftime("%H-%M-%S"))

        with open(f"{self.__cli_arguments['output_dir']}/{self.groname}", mode='w') as gro:
            gro.write(gro_header)
            gro.write(str(len(self.__particles)) + "\n")
            
            if centering == True:
                logging.info("Centering of System coordinates is enabled!")
                #take coordinates from translated (centered) dict
                for atom_index, atom_data in self.centered_system_coords.items():
                    atomname = atom_data["name"]
                    coords = atom_data["coords"]
                    x, y, z = coords
                    #look for atomnames beginning with MX, this is Matrix res
                    if re.match(r'^MX[1-5]$', atomname):
                        resname = "MX  "
                    else:
                        # rest will be CELL
                        resname = "CELL"
                    
                    # resid increases per every increase in the number of particles per cell
                    if atom_index % nr_of_particles == 0:
                        resid += 1

                    atom_number = atom_index + 1
                    
                    # Format the coordinates in GRO format
                    line = "{0:>5}{1:<5}{2:>5}{3:>5}{4:>8.3f}{5:>8.3f}{6:>8.3f}\n".format(
                        resid, resname, atomname, atom_number, x, y, z
                    )

                    gro.write(line)
                    
            if centering == False:
                logging.info("Centering of System coordinates is disabled!")
                #take coordinates from non-translated dict
                for atom_index, (coord, atomname) in enumerate(zip(coords, atomnames)):
                    atomname = atomname
                    x, y, z = coord
                    #look for atomnames beginning with MX, this is Matrix res
                    if re.match(r'^MX[1-5]$', atomname):
                        resname = "MX  "
                    else:
                        # rest will be CELL
                        resname = "CELL"
                    atom_number = atom_index + 1
                    
                    # Format the coordinates in GRO format
                    line = "{0:>5}{1:<5}{2:>5}{3:>5}{4:>8.3f}{5:>8.3f}{6:>8.3f}\n".format(
                        atom_number, resname, atomname, atom_number, x, y, z
                    )

                    gro.write(line)
            print(f"Box size: {extra_box_size}")
            gro.write("{:>10.5}{:>10.5}{:>10.5}\n".format(float(extra_box_size[0]), float(extra_box_size[1]), float(extra_box_size[2])))
        gro.close()
        
        logging.info(f"Built a .gro file '{self.__cli_arguments['output_dir']}/{self.groname}' of the final requested system ")
        print(f"INFO: Coordinate generation finished, '{self.__cli_arguments['output_dir']}/{self.groname}' is saved. ")
        
        #self.construct_system_topology()
        logging.info(f"Topology and subtopologies '{self.__cli_arguments['output_dir']}/system.top' built of the final system ")
        print(f"INFO: All input creation is completed. ")
        logging.info(f"Number of particles in GRO file: {len(self.particles)}")

    def create_topology(self):
        '''Outputs: SYSTEM-{timeprint}.itp file on disk in output dir. '''
        
        now = datetime.datetime.now()
        self.itpname = "SYSTEM-{}.itp".format(now.strftime("%H-%M-%S"))
        
        output_dir_path = self.__cli_arguments['output_dir']
        if not os.path.exists(f"{output_dir_path}/toppar"):
            os.makedirs(f"{output_dir_path}/toppar")         
        
        with open(f"{self.__cli_arguments['output_dir']}/toppar/{self.itpname}", "w") as itp:
            #Write the topology header
            header = "; Topology file for a system generated at {}\n".format(now.strftime("%H:%M:%S"))
            itp.write(header)
            ff_itp =  f"; At the same time, copied the force field givend by -ff-dir into 'toppar' from:\n;#include \"{self.__ff_parser.itp_path}\"\n"
            itp.write(ff_itp)
            #we copy the force field so the force field stays constant for this simulation and we can 
            # trust the results match with the topology
            shutil.copy2(self.__ff_parser.itp_path, f"{self.__cli_arguments['output_dir']}/toppar/forcefield.itp")

            coords = [particle.get_coordinates() for particle in self.__particles]
            atomnames = [particle.get_type() for particle in self.__particles]
            cell_ids = [cell.cell_id for cell in self.__cells for particle in cell.get_surface_particles()]

            # Write the [moleculetype] directive and the [atoms] directive based on the atomnames dict
            itp.write("\n[ moleculetype ]\n; Name        nrexcl\n  CELL        1\n\n[ atoms ]\n; nr type resnr residue atom cgnr  charge   mass\n")
            for i, (coord, atomname, cell_id) in enumerate(zip(coords, atomnames, cell_ids)):  
                resname = "CELL"
                atom_nr = i + 1 
                mass = self.__ff_parser.atomtypes[atomname]['mass']
                itp.write(f'  {atom_nr:<3d}  {atomname:<3s}   {cell_id:<3d}    {resname:<3s}    {atomname:<3s}  {atom_nr:<3d} 0.0000  {str(mass):<3s}\n')
                #itp.write("  {:<3s}  {:<3s}   1    {:<3s}    {:<3s}  {:<3s} 0.0000  {:<3s}\n".format(str(atom_nr), atomname, resname, atomname, str(atom_nr), str(mass)))

            # writing [bonds] directive
            # First, connect all membrane beads to the center based on their atomnames. That is, each atom is connected to C bead (first atom)
            # do this by comparing the self.atomnames dict to the bondtype dict read from the force field and look for bonds with a C bead in them
            itp.write("\n[ bonds ]\n; i j func   r0   fk\n; center - membrane bonds\n")
            
            for i, (coord, atomname) in enumerate(zip(coords, atomnames)):
                if atomname != 'C': #skip the center bead
                    parsed = False #because of the two if statements required in both dicts, the for loop prints multiple times. Workaround (I am a genius coder ;D)
                    for bond, entry in self.__ff_parser.bondtypes.items():
                        if "C" in bond and atomname in bond:
                            if not parsed:
                                atom_index = i + 1
                                #by construction, the Center bead is indexed as 1
                                itp.write(" 1  {:<3s} {:<3s} {:<3s} {:<3s} \n".format(str(atom_index), str(entry['func']), str(entry['r0']), str(entry['fk'])))
                                parsed = True

            #Finally, we take the self.nnneighbour dict and use it to parse those bonds separately
            
            neighbours_dict = dict((k, v) for cell in self.__cells for k, v in cell.nnneighbours.items())

            # first check if the dict is not empty (that would mean the setting was disabled!)
            if not neighbours_dict:
                pass
            
            else:
                itp.write("; membrane surface neighbour bonds\n")
                
                #neighbours are constructed per particle, so there will be duplicate bonds as a result. 
                # imagine e.g. '3 8' vs '8 3' --> these are functionally identical
                existing_bonds = set() #Store only unique bonds here
                

                for index_atomname, neighbours in neighbours_dict.items():
                    #extract the index and atom name of each surface particle in the system
                    atom_index, atomname= index_atomname.split() #str datatype, so need to split it
                    for neighbour in neighbours:
                        neighbour_atom_index, neighbour_atom_name = neighbour
                        # Sort the indices to avoid duplicates (basically, '2 3' and '3 2' are considered the same)
                        bond_indices = tuple(sorted([int(atom_index), int(neighbour_atom_index)]))
                        if bond_indices not in existing_bonds:
                            existing_bonds.add(bond_indices)
                            #look for the bonded type based on either order of the atomnames
                            bond_type = self.__ff_parser.bondtypes.get((atomname, neighbour_atom_name)) or self.__ff_parser.bondtypes.get((neighbour_atom_name, atomname))
                            #and extract the bonded information for that bonded pair
                            if bond_type:
                                itp.write(" {:<3s} {:<3s} {:<3s}   {:<3s}   {:<3s} \n".format(str(bond_indices[0]), str(bond_indices[1]), str(bond_type['func']), str(bond_type['r0']), str(bond_type['fk'])))
                            else:
                                logging.warning(f"Bondtype not found in the force field for atom index {atom_index} and {neighbour_atom_index}")

        itp.close()
        logging.info(f"Built a cell topology file '{self.__cli_arguments['output_dir']}{self.itpname}' of a single cell. ")
        logging.info(f"Number of particles in ITP file: {len(coords)}")
        print(f"INFO: Topology generation finished, '{self.__cli_arguments['output_dir']}{self.itpname}' is saved. ")
        
        if self.__cli_arguments['verbose']:
            print(f"\nVERBOSE MODE ENABLED. A cell topology file '{self.__cli_arguments['output_dir']}/{self.itpname}' has been built")
    

    def construct_system_topology(self):
        """
        Constructs the top file required to run the simulation, links itps and forcefield
        """
               
        matrix_on_off = self.__json_values["matrix_on_off"]
        
        with open(f"{self.__cli_arguments['output_dir']}/system.top", "w") as top:
            forcefield =  f"#include \"{self.__cli_arguments['output_dir']}toppar/forcefield.itp\"\n"
            top.write(forcefield)
            cell_itp = f"#include \"{self.__cli_arguments['output_dir']}toppar/{self.itpname}\"\n"
            top.write(cell_itp)
            #if matrix_on_off == 'on':      # not implemented yet
                # matrix_itp = f"#include \"{self.__cli_arguments['output_dir']}/toppar/{self.matrix.itpname}\"\n"
                # top.write(matrix_itp)
            
            number_of_cells = self.__json_values["number_of_cells"]
            top_info =  f"\n[ system ]\nCellSimGMX system\n\n[ molecules ]\nCELL {number_of_cells}" 
            top.write(top_info)
            #the matrix individual particles are all written to .itp because of posres so there is only
            # a single matrix 'molecule' in the simulation
            matrix = "\nMX      1"
            if matrix_on_off == 'on':
                top.write(matrix)
        top.close()