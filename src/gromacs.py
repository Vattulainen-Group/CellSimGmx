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

class Simulation:
    '''Prepares mdp files and runs the simulation'''
    def __init__(self, cli_arguments: dict, ff_parser: ForcefieldParserGMX, json_values: dict):
        self.__cli_arguments = cli_arguments
        self.__ff_parser = ff_parser
        self.__json_values = json_values

    def write_mdp_minim(self):
        """Simple function to construct mdp of minimization routine, based on JSON input"""
        
        minimization = self.__json_values["minimization"]
        
        steep_mdp = {
            #basic minimization routine
            'integrator': 'steep',
            'nsteps': '10000',
            'emtol': '10',
            'emstep': '0.01',
            'nstxout-compressed': '1000',
            'cutoff-scheme': 'Verlet',
            'vdw_type': 'cutoff',
            'vdw-modifier': 'Potential-shift-verlet',
            'rvdw': '1.1',
            'rcoulomb': '1.1',
            #don't have electrostatics in system so can ignore all electrostatics .mdp settings
            }
        
        cg_mdp = {
            # more stringent minimization to deal with poor initial configurations
            'integrator': 'cg',
            'nsteps': '10000',
            'emtol': '5',
            'emstep': '0.01',
            'nstxout-compressed': '1000',
            'cutoff-scheme': 'Verlet',
            'vdw_type': 'cutoff',
            'vdw-modifier': 'Potential-shift-verlet',
            'rvdw': '1.1',
            'rcoulomb': '1.1',
            #don't have electrostatics in system so can ignore all electrostatics .mdp settings     
        }
        
        em_dict = steep_mdp if minimization == "steep" else cg_mdp

        with open(f"{self.__cli_arguments['output_dir']}em.mdp", mode='w') as mdp:
            mdp.write(f"; {minimization.capitalize()} descent minimization\n")
            for key, value in em_dict.items():
                mdp.write(f"{key} = {value}\n")
