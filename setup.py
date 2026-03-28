from setuptools import find_packages,setup
from typing import List

HYPHEN_E_DOT =  "-e ."

def get_req(filepath:str)->List[str]:
    req = [];
    with open(filepath) as file:
        req = file.readlines()
        req = [item.replace("\n", " ") for item in req]

        if HYPHEN_E_DOT in req:
            req.remove(HYPHEN_E_DOT)
        
    return req



setup(
    name="mlproject",
    version="0.0.1",
    author="Aaryan",
    author_email="aaryan.sharda31@gmail.com",
    packages=find_packages(),
    install_requires=get_req('requirements.txt'),
)