# The setup script is the center of all activity in building, distributing, and installing modules using the Distutils.
# The main purpose of the setup script is to describe your module distribution to the Distutils,
# so that the various commands that operate on your module distribution will do the right thing.

#responsible in creating machine learning application as a package 
# and then deploy that as well. 

from setuptools import find_packages , setup# find_packages is a function that automatically discovers all packages and subpackages within a directory.
from typing import List 

def get_requirements(file_path:str)->List[str]:
    '''
    This function will return the list of requirements
    '''
    requirements = []
    with open('requirements.txt') as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n","") for req in requirements]
        
        if '-e .' in requirements:
            requirements.remove('-e .')
    return requirements
            
setup(
    name= "ml_project" ,
    version= "0.0.1" ,
    author= "Muhammad Obaidullah" ,
    author_email= "obaidlgs2005@gmail.com",
    packages= find_packages() ,
    install_requires= get_requirements('requirements.txt') 
)