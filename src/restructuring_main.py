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
from gromacs import Simulation
from system import Particle
from system import Cell
from system import Layer
from system import System

import config

def main():
    # call parsers
    cli_parser = CLIParser()
    cli_arguments = cli_parser.parse_args()

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

    sim = Simulation(cli_arguments, ff_parser, values)
    sim.write_mdp_minim()

if __name__ == "__main__":
    main()
