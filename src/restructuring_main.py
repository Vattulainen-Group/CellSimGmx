import numpy as np
import matplotlib.pyplot as plt
import datetime
import logging
from scipy.spatial import KDTree    #for n-nearest neighbour search
import os
import shutil

from settings_parser import CLIParser
from settings_parser import JSONParser
from settings_parser import ForcefieldParserGMX
from gromacs import Simulation, ExecuteSimulations
from system import Particle
from system import Cell
from system import Layer
from system import System

import config

def main():
    ### LOGGING DETAILS
    ### note, logfile is saved where programme is executed, change this to output directory at some point if possible
    now = datetime.datetime.now()
    logging.basicConfig(
    filename = "cellsimgmx-{}.log".format(now.strftime("%H-%M")),
    level=logging.INFO, #print >INFO msgs to logfile
    format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    terminal_handler = logging.StreamHandler()
    terminal_handler.setLevel(logging.WARNING)  # Only print WARNINGS or ERRORS to terminal
    terminal_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

    logging.getLogger('').addHandler(terminal_handler)
    
    log_handler = None
    for handler in logging.getLogger('').handlers:
        if isinstance(handler, logging.FileHandler):
            #to obtain log file name
            log_handler = handler
            break
    #### END LOGGING DETAILS
    logging.info("Started programme execution. ")

    # call parsers
    cli_parser = CLIParser()
    cli_arguments = cli_parser.parse_args()
    print(cli_arguments)

    json_parser = JSONParser(cli_arguments)
    json_parser.load_json()
    values = json_parser.extract_json_values()

    ff_parser = ForcefieldParserGMX(cli_arguments)
    ff_parser.parse_GMX_ff()

    # create system
    system = System(values, cli_arguments, ff_parser)
    system.create_system()
    system.build_gro_file_system()
    system.create_topology()
    system.construct_system_topology()

    # simulation preparation
    sim = Simulation(cli_arguments, ff_parser, values)  # creating an instance creates minimisation, equilibration and production mdp files
    sim.write_mdp_minim()
    sim.write_mdp_eq_prod()
    execute_sims = ExecuteSimulations()

    print(f"CellSimGMX finished. '{log_handler.baseFilename}' has been saved.")

if __name__ == "__main__":
    main()
