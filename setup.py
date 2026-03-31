from setuptools import find_packages, setup
from typing import List

HYPEN_E_DOT = '-e .'

def get_requirements(file_path: str) -> List[str]:
    '''
    This function will return the list of requirements
    '''
    requirements = []
    try:
        with open(file_path) as file_obj:          # ✅ Fix 1: use file_path parameter
            requirements = file_obj.readlines()
            requirements = [req.strip() for req in requirements]
            
            if HYPEN_E_DOT in requirements:
                requirements.remove(HYPEN_E_DOT)
    except FileNotFoundError:
        print("requirements.txt file not found.")
        
    return requirements

setup(
    name="ml_project",
    version="0.0.1",
    author="Muhammad Obaidullah",
    author_email="obaidlgs2005@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)