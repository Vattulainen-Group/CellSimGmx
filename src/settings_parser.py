# -*- coding: utf-8 -*-
# Copyright (C) 2024 Mats Punt mats.punt(at)helsinki.fi
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.

import argparse
import logging
import json
import glob
import sys
import os
import re

class CLIParser:
    """
    This class accepts command line arguments of the programme and stores them in a dictionary. 
    """
    def __init__(self):
        self.parser = argparse.ArgumentParser(description="CellSimGMX: A 3D discrete element framework simulator for epithelial tissues using GROMACS")
        
        # specifying directories for programme
        # the reason for using directories instead of direct files is to support loading in multiple settings and force field files in case of parallel simulations
        self.parser.add_argument("--ff-dir", "-ff", help="Path to the directory where the forcefield is stored (accepts arbitrary .itp file names)")
        self.parser.add_argument("--input-dir", "-in", help="Path to the directory where the simulation input is stored (accepts arbitrary .json file name)")
        self.parser.add_argument("--output-dir", "-out", help="Path where files should be generated and simulation should be run")
        
        #Simulation related
        self.parser.add_argument("--no-sim", "-nosim", action='store_true', help="Does not run a simulation, but does generate all input files like a dryrun.")
        
        #Miscellaneous
        self.parser.add_argument("--verbose", "-v", action="store_true", help="Optional argument. When enabled, prints detailed logging. Useful for debugging output. ")
        self.__arguments = {}

    def parse_args(self):
        args = self.parser.parse_args()
        for arg in vars(args):
            self.__arguments[arg] = getattr(args, arg)
        return self.__arguments

class JSONParser:
    """
    This class handles parsing of the input.JSON file. An instance of this class can be passed to other classes to access the 
    simulation settings. Currently supports four classes of settings "Cell", "Tissue", "Matrix" and "Simulation", as seen in the example input.JSON. 
    
    Attributes:
        cli_parser (CLIParser): An instance of the CLIParser class
        json_path (str): The path to the JSON input file.
        data (dict): The loaded JSON data.
    """
    
    def __init__(self, cli_arguments):
        self._cli_arguments = cli_arguments
        self.json_path = None
        self.data = None
        self.json_values = None
        
    def load_json(self):
        """
        Loads the input.json as data, does a bunch of checks the input is OK, verifies input complies with JSON standard. 
       
        Params:
            args.input_dir (str) - Location of the directory where the JSON file should be loaded from. 
        """
        
        if self._cli_arguments['input_dir']:
            if not os.path.exists(self._cli_arguments['input_dir']):
                logging.error(f"Error: Input dir '{self._cli_arguments['input_dir']}' does not exist. Please check for typos and try again")
                sys.exit(1)
                
            if not os.access(self._cli_arguments['input_dir'], os.R_OK):
                logging.error(f"Error: Input directory '{self._cli_arguments['input_dir']}' is not readable. Do you have permissions and is the directory correct?")
                sys.exit(1)

        else:
            logging.error("Error: Input dir for settings not specified. Please specify using '--input-dir' and try again")
            sys.exit(1)

        json_files = glob.glob(f"{self._cli_arguments['input_dir']}/*.json")
        
        if len(json_files) > 1:
            logging.error("Multiple .JSON files were found in your input dir which is causing conflicts. Remove the incorrect ones and try again: \n") 
            for i in json_files:
                logging.error(str(i))
            sys.exit(1)
            
        if len(json_files) > 0: 
            self.json_path = json_files[0]
            try:
                with open(self.json_path, 'r') as file:
                    self.data = json.load(file)
                    logging.info(f"Using simulation input from: '{self.json_path}'")
            except json.JSONDecodeError:
                logging.error("The supplied JSON file contains formatting errors, please check it complies with the template supplied with the CellSimGMX package and correct any typos")
                sys.exit(1)
            except FileNotFoundError:
                logging.warning("No input.JSON could be found in your '--input-dir' entry. Are you in the right directory and is your file extension correct?")
                sys.exit(1)
        else:
            logging.warning("No input.JSON could be found in your '--input-dir' entry. Are you sure you supplied a correct path?")
            sys.exit(1)

    def extract_json_values(self):
        """
        Extracts values from input.JSON and stores them in variables. Variable names are by construction IDENTICAL to JSON variable names. 
        e.g. "simulation_type = 'cell'" in both JSON and CellSimGMX. 
        
        Returns:
            json_values (dict) - A dict containing all variables and their values required as CellSimGMX input
        """
        
        if self.data is None:
            logging.error("Some unknown problem occurred reading the input.JSON. Terminating programme...")
            sys.exit(1)

        json_values = {}
        for _, category_items in self.data.items():
            for key, value_data in category_items.items():
                value = value_data.get("value")
                if value is not None:
                    var_name = f"{key}".lower()
                    json_values[var_name] = value

                    # Do some basic checks on the input to guide the user
                    if var_name == "number_of_cells":
                        if not (isinstance(value, int) and 1 <= value <= 2000):
                            logging.warning(f"You have specified {value} cells in the system, this is possible but will significantly slow down simulations!")

                    if var_name == "initial_packing_shape":
                        if not (isinstance(value, str) and value in ['spherical', 'cuboid', 'ellipsoid']):
                            logging.error(f"{value} is not a valid cell shape!")
                            sys.exit(1)

                    if var_name == "tissue_packing":
                        if not (isinstance(value, str) and value in ['grid', 'hexagonal', 'monolayer', 'disordered']):
                            logging.error(f"{value} is not a valid packing for tissues!")
                            sys.exit(1)
                         
                    if var_name == "membrane_beads":
                        if not isinstance(value, str) and not all(re.match(r'^M[1-5]$', part.strip()) for part in value.split(",")):
                            logging.error(f"{value} is an incorrect option for junction_beads!")
                            sys.exit(1)
                            
                    if var_name == "junction_beads":
                        if not isinstance(value, str) and value == "off" and not all(re.match(r'^J[1-5]$', part.strip()) for part in value.split(",")):
                            logging.error(f"{value} is an incorrect option for membrane_beads!")
                            sys.exit(1)
                    
                    if var_name == "ensemble":
                        if not (isinstance(value, str) and value in ['NVE', 'NVT', 'NpT']):
                            logging.error(f"{value} is not a valid simulation ensemble!")
                            sys.exit(1)
                            
                    if var_name == "minimization":
                        if not (isinstance(value, str) and value in ['steep', 'cg']):
                            logging.error(f"{value} is not a valid minimization option!")
                            sys.exit(1)
                            
                    
                                                    
        if self._cli_arguments['verbose']:
            print(f"\nVERBOSE MODE ENABLED. Parameters read from: {self.json_path}.") 
            print("These are the parameters you have configured\n")
            for var_name, value in json_values.items():
                print(f"{var_name} : {value}")

        # Always save parameter information to the logfile
        logging.info("CellSimGMX has found the following parameters to be configured:\n")
        for var_name, value in json_values.items():
            logging.info(f"{var_name} : {value}")

        self.json_values = json_values #store the JSON values in the Class
        return json_values

class ForcefieldParserGMX:
    """
    This class handles parsing of the predefined force field in GROMACS format and stores it in dicts

    Attributes:
        atomtypes (dict)
        nonbond_params (dict)
        bondtypes (dict)
    """
    
    def __init__(self, cli_arguments):
        self._cli_arguments = cli_arguments
        self.atomtypes = {}
        self.nonbond_params = {}
        self.bondtypes = {}
        self.itp_path = None

    def parse_GMX_ff(self):
        """
        Looks for a GMX compatible .itp file in the user-specific directory. Then uses helper functions to parse the 
        different .itp entries into dictionaries. 
            
        Raises:
            - SystemExit: Raised with an error message and exit status 1 if there are issues with the force field directory
                or if multiple .itp files are found.
            - FileNotFoundError: Raised if no .itp files could be found in the specified force field directory.
        """

        if self._cli_arguments['ff_dir']:
            if not os.path.exists(self._cli_arguments['ff_dir']):
                logging.error(f"Force field directory '{self._cli_arguments['ff_dir']}' does not exist. Please check for typos and try again")
                sys.exit(1)
            
            if not os.access(self._cli_arguments['ff_dir'], os.R_OK):
                logging.error(f"Force field directory '{self._cli_arguments['ff_dir']}' is not readable. Do you have permissions and is the directory correct?")
                sys.exit(1)
        else:
            logging.error("Force field directory not specified. Please specify using '--ff-dir' and try again ")
            sys.exit(1)

        itp_files = glob.glob(f"{self._cli_arguments['ff_dir']}/*.itp")

        if len(itp_files) > 1:
            logging.error("Multiple .itp files were found in the force field directory which is causing conflicts. Aborting run.")
            print("Remove the incorrect files and please try again:")
            for i in itp_files:
                print(str(i))
            sys.exit(1)

        if len(itp_files) > 0:
            self.itp_path = itp_files[0]
            try:
                with open(self.itp_path, 'r') as file:
                    self.itp_data = file.read()
                logging.info(f"Using force field input from: {self.itp_path}")
            except FileNotFoundError:
                logging.error("No .itp files could be found in your '--ff-dir' entry. Are you in the right directory and is your file extension correct?")
        else:
            logging.error("No .itp files could be found in your '--ff-dir' entry. Are you in the right directory and is your file extension correct?")

        # open the force field .itp and extract the different '[ ]' categories
        # it makes little sense to check the formatting here ( I would also not know how to do it systematically?) as the user 
        # does not need to edit the force field themselves. But have to be careful of mistakes creeping in the force field!
        with open(self.itp_path, 'r') as itp:
            section = None
            for line in itp:
                line = line.strip()
                if line.startswith('[') and line.endswith(']'):
                    section = line[1:-1].strip()
                elif section == 'atomtypes':
                    self.parse_atomtype_GMX_helper(line)
                elif section == 'nonbond_params':
                    self.parse_nonbond_params_GMX_helper(line)
                elif section == 'bondtypes':
                    self.parse_bondtype_GMX_helper(line)
        
        if self._cli_arguments['verbose']:
            atom_names = list(self.atomtypes.keys())
            print(f"\nVERBOSE MODE ENABLED. Succesfully read the force field from {self.itp_path}. ")
            print("The force field contains the following particle types:")
            print(atom_names)
            
            print("\nAnd the following LJ interaction pairs:")
            interaction_pairs = list(self.nonbond_params.keys())
            print(interaction_pairs)
            
            print("\nAnd the following bonded types:")
            bonded_types = list(self.bondtypes.keys())
            print(bonded_types)

        logging.info("CellSimGMX has found the following atom types in the specified force field:")
        atom_names = list(self.atomtypes.keys())
        logging.info(atom_names)
        
        logging.info("And the following LJ interaction pairs:")
        interaction_pairs = list(self.nonbond_params.keys())
        logging.info(interaction_pairs)
        
        logging.info("And the following bonded types:")
        bonded_types = list(self.bondtypes.keys())
        logging.info(bonded_types)

    def parse_atomtype_GMX_helper(self, line):
        """
        Parses atom types from the .itp so .gro files can be built from them. 
            
        Returns:
            self.atomtypes (dict) - A dict with key = atomtypes and values callable as .itp variables (e.g. 'mass')
        """
        if not line.startswith(';'):
            columns = line.split()
            if len(columns) >= 6:
                atomtype = columns[0]
                mass = float(columns[1])
                charge = float(columns[2])
                ptype = columns[3]
                sigma = float(columns[4])
                epsilon = float(columns[5])
                self.atomtypes[atomtype] = {'mass': mass, 'charge': charge, 'ptype': ptype, 'sigma': sigma, 'epsilon': epsilon}

    def parse_nonbond_params_GMX_helper(self, line):
        """
        Parses nonbonded (Lennard-Jones) types in the .itp.

        Returns:
            self.nonbond_params (dict) - A dict with key = [i,j] where i and j are two interacting atom types.
                
        Note:
            This function and nonbonded information is currently not used for anything. Maybe for M-C routine later?
        """
        #skip lines with comments
        if not line.startswith(';'):
            columns = line.split()
            if len(columns) >= 5:
                i = columns[0]
                j = columns[1]
                func = int(columns[2])
                sigma = float(columns[3])
                epsilon = float(columns[4])
                self.nonbond_params[(i, j)] = {'func': func, 'sigma': sigma, 'epsilon': epsilon}

    def parse_bondtype_GMX_helper(self, line):
        """
        Parses bonded types in the force field .itp. 

        Returns:
            self.bonded_types (dict) - A dict with key = [i,j] where i and j are two bonded atom types (e.g. M1 & M2.)
                
        Note:
            This function and nonbonded information is currently not used for anything. Maybe for M-C routine later?
        """
        #skip lines with comments
        if not line.startswith(';'):
            columns = line.split()
            if len(columns) >= 5:
                i = columns[0]
                j = columns[1]
                func = int(columns[2])
                r0 = float(columns[3])
                fk = float(columns[4])
                self.bondtypes[(i, j)] = {'func': func, 'r0': r0, 'fk': fk}


