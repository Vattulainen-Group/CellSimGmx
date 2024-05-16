import numpy as np
import matplotlib.pyplot as plt
import datetime
import logging
import subprocess as sp
import os
import shutil
import re
import sys
from distutils.spawn import find_executable
import glob

from settings_parser import CLIParser
from settings_parser import JSONParser
from settings_parser import ForcefieldParserGMX

import config

class Simulation:
    '''Prepares mdp files and runs the simulation. Ensures and checks all necessary files are there to run GMX simulations. Creates .mdp / runfiles based
    on input.JSON for the minimization, equilibration and production part of the simulation. 
    
    write_mdp_minim():
        Creates mdp file of minimization routine based on input.JSON
        
    write_mdp_eq_prod():
        Creates mdp file of equilibration and production routine based on input.JSON '''
    
    def __init__(self, cli_arguments: dict, ff_parser: ForcefieldParserGMX, json_values: dict):
        self.__cli_arguments = cli_arguments
        self.__ff_parser = ff_parser
        self.__json_values = json_values

        self.gmx_path = None
        if self.__cli_arguments['no_sim']:
            logging.warning("You disabled simulations using the '--no-sim' flag, exiting now...")
            sys.exit(1)

        else:
            print("INFO: Starting simulation routine. ")
            logging.info(f"Succesfully made it to the simulation routine")
            self.gmx_path = find_executable("gmx") #check whether gmx is available in the path and print to the user.
            if self.gmx_path is not None:
                print(f"INFO: GROMACS found at: '{self.gmx_path}'")
                logging.info(f"Found a GROMACS executable at: '{self.gmx_path}'")
            else:
                logging.error("GROMACS executable not found. Please make sure it's installed and sourced in PATH as 'gmx' before trying again")
                sys.exit(1)
                        
        # self.write_mdp_minim()
        # self.write_mdp_eq_prod()

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
        path = os.path.join(self.__cli_arguments['output_dir'], 'mdps')

        if not os.path.exists(path):
            os.mkdir(path)
        else:
            shutil.rmtree(path)
            os.mkdir(path)

        with open(f"{self.__cli_arguments['output_dir']}mdps/em.mdp", mode='w') as mdp:
            mdp.write(f"; {minimization.capitalize()} descent minimization\n")
            for key, value in em_dict.items():
                mdp.write(f"{key} = {value}\n")

    def write_dict_mdp_helper(self, dict, mdp_name):
        """Helper function to write dictionary contents to MDP file

        Args:
            dictionary (dict): Dict containing MDP options
            filename (str): Name of the mdp_file you want to write to
        """
        with open(f"{self.__cli_arguments['output_dir']}{mdp_name}", mode='w') as file:
            for key, value in dict.items():
                file.write(f"{key} = {value}\n")

    def write_mdp_eq_prod(self):
        """
        Writes equilibration and production MDP files based on requested simulation routine
        """
        
        ensemble = self.__json_values["ensemble"]
        timestep = self.__json_values["timestep"] #sets production only!
        number_of_steps = self.__json_values["number_of_steps"]
        number_of_steps = tuple(map(int, number_of_steps.split(', '))) # [0] is equilibration nsteps, [1] is production nsteps
                
        def extend_mdp(base_mdp, extra_mdp_options):
            return {**base_mdp, **extra_mdp_options}

        NVE_mdp = {
            #basic NVE mdp, only options which are different from default GMX settings are included
            'integrator': 'md',
            'dt': '0.02', # timestep for equilibration
            'nsteps': number_of_steps[0], # sets number of steps for equilibration
            'nstxout-compressed': '1000',
            'compressed-x-precision': '100',
            'nstlist': '20',
            'ns_type': 'grid',
            'pbc': 'xyz',
            'vdw_type': 'cutoff',
            'vdw-modifier': 'Potential-shift-verlet',
            'rvdw': '1.1',
            'rcoulomb': '1.1',
            'constraints': 'none'
            # tcoupl and pcoupl are disabled by default (don't need to specify)
        }

        self.write_dict_mdp_helper(NVE_mdp, os.path.join("mdps/NVE_eq.mdp"))

        #EQUILIBRATION with Berendsen coupling
        NVT_eq_mdp = extend_mdp(NVE_mdp, {'tcoupl': 'berendsen', 'tc-grps': 'System', 'tau-t': '1.0', 'ref-t': '310'})
        NpT_eq_mdp = extend_mdp(NVT_eq_mdp, {'pcoupl': 'berendsen', 'pcoupltype': 'isotropic', 'tau_p': '2.0', 'ref_p': '1.0', 'compressibility': '4.5e-05', ';refcoord-scaling': 'com'}) #refcoord-scaling is required for simulations with pressure coupling and position restraints

        #PRODUCTION with v-rescale thermostat and c-rescale barostat
        NVE_mdp['dt'] = timestep
        NVE_mdp['nsteps'] = number_of_steps[1]
        NVT_prod_mdp = extend_mdp(NVE_mdp, {'tcoupl': 'v-rescale', 'tc-grps': 'System', 'tau-t': '1.0', 'ref-t': '310'})
        NpT_prod_mdp = extend_mdp(NVT_prod_mdp, {'pcoupl': 'C-rescale', 'pcoupltype': 'isotropic', 'tau_p': '5.0', 'ref_p': '1.0', 'compressibility': '4.5e-05', ';refcoord-scaling': 'com'})
        
        #Save required mdps dependent on requested ensemble
        if ensemble == "NVE":
            NVE_mdp['dt'] = timestep
            NVE_mdp['dt'] = number_of_steps[1]
            self.write_dict_mdp_helper(NVE_mdp, "mdps/production.mdp")

        elif ensemble == "NVT":
            self.write_dict_mdp_helper(NVT_eq_mdp, "mdps/NVT_eq.mdp")
            self.write_dict_mdp_helper(NVT_prod_mdp, "mdps/production.mdp")

        elif ensemble == "NpT":
            self.write_dict_mdp_helper(NVT_eq_mdp, "mdps/NVT_eq.mdp")
            self.write_dict_mdp_helper(NpT_eq_mdp, "mdps/NpT_eq.mdp")
            self.write_dict_mdp_helper(NpT_prod_mdp, "mdps/production.mdp")


class GromppWrapper:
    """
    Wrapper to generate .tpr files using gromacs (gmx)
    """
    def __init__(self, topol, coord, mdp, output_tpr, maxwarn=0):
        self.topol = topol
        self.coord = coord
        self.mdp = mdp
        self.output_tpr = output_tpr
        self.maxwarn = maxwarn
        self.run_grompp()

    def run_grompp(self):

        grompp = [
            'gmx', 'grompp', #already checked for gmx executable before, for now assuming this is available
            '-p', self.topol,
            '-c', self.coord,
            '-f', self.mdp,
            '-o', self.output_tpr,
            '-maxwarn', str(self.maxwarn)
        ]

        try:
            grompp_result = sp.run(grompp, stdout=sp.PIPE, stderr=sp.PIPE, text=True, check=True) #grompp output is piped, only shown if execution fails
            print(f"INFO: Succesfully preprocessed gmx files (grompp) as '{self.output_tpr}'")
            logging.info(f"Succesfully preprocessed gmx files (grompp) as '{self.output_tpr}'")
        except sp.CalledProcessError as e:
            logging.error(f"There was an error running the command '{' '.join(e.cmd)}'")
            grompp_error = e.stderr.splitlines()
            for line in grompp_error[-15:]:
                print(line)  #print final 15 lines of grompp error for user to troubleshoot
            sys.exit(1)

        return grompp_result.stdout
            
class MDRunWrapper:
    """Wrapper to run gromacs (gmx) simulations. 
    
    There are more mdrun options that can be added later, for instance multidir could be used, ntomp, ntmpi etc. 
    """
    def __init__(self, tpr_name, nthreads=4): 
        self.tpr_name = tpr_name
        self.nthreads = nthreads
        self.run_mdrun()
        
    def run_mdrun(self):
        mdrun = ['gmx', 'mdrun', '-v', '-deffnm', self.tpr_name, '-nt', str(self.nthreads)]

        try:
            mdrun_process = sp.Popen(mdrun, stdout=sp.PIPE, stderr=sp.PIPE, text=True)
            
            while True:
                output_line = mdrun_process.stderr.readline()
                if not output_line and mdrun_process.poll() is not None:
                    break
                # this is a bit buggy but it is meant to print only a single line at a time of the mdrun output --> useful for tracking of simulation duration
                sys.stdout.write(f"\r{output_line.strip()}")
                sys.stdout.flush()
            sys.stdout.flush()
            
            mdrun_process.wait()
            print(f"INFO: Succesfully completed mdrun request '{self.tpr_name}'            ")
            logging.info(f"Succesfully completed mdrun request '{self.tpr_name}'")

        except sp.CalledProcessError as e:
            logging.error(f"There was an error running the command '{' '.join(e.cmd)}' \n")
            mdrun_error = e.stderr.splitlines()
            for line in mdrun_error[-15:]:
                print(line)  # print final 15 lines of mdrun error for the user to troubleshoot
            sys.exit(1)

        return mdrun_process.returncode == 0

class ExecuteSimulations:
    """
    In this class, all required simulations are grompped and executed. The required simulations are inferred from the mdps that are saved in the class "SimulationPreparation")). 
    """ 
        
    def __init__(self):   
        self.sim_list = glob.glob("mdps/*mdp")
        self.init_coord = glob.glob(f'SYSTEM*.gro')[0] if glob.glob(f'SYSTEM*.gro') else None
        self.topol = "system.top" #this is hardcoded anyway
        self.run_simulations()
         
    def get_mdp_names(self, mdp_dir='mdp'):
        """Based on the mdp names, the simulations will be executed. 

        Args:
            mdp_dir (str): Path where mdp files are stored
        """
        self.sim_list = glob.glob(os.path.join(mdp_dir, '*.mdp'))
        
    def run_simulations(self):
        
        def create_directory(sim_path):
            if not os.path.exists(sim_path):
                os.makedirs(sim_path)
                     
        #minimization is always required
        create_directory("em")
        GromppWrapper(self.topol, self.init_coord, "mdps/em.mdp", "em/em.tpr")
        MDRunWrapper("em/em")
        
        #NVE equilibration is always required
        create_directory("NVE_eq")
        GromppWrapper(self.topol, "em/em.gro", "mdps/NVE_eq.mdp", "NVE_eq/NVE_eq")
        MDRunWrapper("NVE_eq/NVE_eq")
        
        #figure out if NVT and NpT equilibration are required
        eq_mdps = [mdp for mdp in self.sim_list if 'eq' in mdp]
        
        if "mdps/NVT_eq.mdp" in eq_mdps:
            create_directory("NVT_eq")
            GromppWrapper(self.topol, "NVE_eq/NVE_eq.gro", "mdps/NVT_eq.mdp", "NVT_eq/NVT_eq", maxwarn=1) #for Berendsen thermostat
            MDRunWrapper("NVT_eq/NVT_eq")
            
        if "mdps/NpT_eq.mdp" in eq_mdps:    
            create_directory("NpT_eq")
            GromppWrapper(self.topol, "NVT_eq/NVT_eq.gro", "mdps/NpT_eq.mdp", "NpT_eq/NpT_eq", maxwarn=2) #for Berendsen therm+barostat
            MDRunWrapper("NpT_eq/NpT_eq")
            
        create_directory("prod")
            
        if "mdps/NVT_eq.mdp" in eq_mdps and "mdps/NpT_eq.mdp" not in eq_mdps:
            GromppWrapper(self.topol, "NVT_eq/NVT_eq.gro", "mdps/production.mdp", "prod/prod")
            #run NVT production
            MDRunWrapper("prod/prod")
            
        if "mdps/NVT_eq.mdp" and "mdps/NpT_eq.mdp" in eq_mdps:
            GromppWrapper(self.topol, "NpT_eq/NpT_eq.gro", "mdps/production.mdp", "prod/prod")
            # run NpT production
            MDRunWrapper("prod/prod")
            
        if "mdps/NVT_eq.mdp" not in eq_mdps:
            GromppWrapper(self.topol, "NVE_eq/NVE_eq.gro", "mdps/production.mdp", "prod/prod")
            # run NVE production
            MDRunWrapper("prod/prod")
            
        #cleanup
        if os.path.exists("mdout.mdp"):
            os.remove("mdout.mdp")