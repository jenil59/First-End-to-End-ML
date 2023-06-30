from setuptools import find_packages,setup

HYPHEN_E_DOT = '-e .'

def get_requirements(file_path):
    """
    this function is just return requirements from file 
    """
    requirements=[]
    with open(file_path) as file:
        requirements = file.readlines()
        requirements = [ req.replace('\n',"") for req in requirements]
        if HYPHEN_E_DOT in requirements:
            requirements.remove(HYPHEN_E_DOT)

    return requirements

get_requirements('requirements.txt')


setup(
   name="FirstMlProject",
   version='0.0.1',
   author='Jenil',
   author_email='jenilsaliya1234@gmail.com',
   packages=find_packages(),
   install_requires=get_requirements('requirements.txt'), 
)