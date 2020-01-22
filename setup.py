from setuptools import find_packages, setup

setup(
    name="agd",
    version="0.0.2",
    packages=find_packages(),
    install_requires=["numpy~=1.0", "scipy~=1.0"],
    author="Jean-Marie Mirebeau",
    author_email="jm.mirebeau@gmail.com",
    description="Adaptive Grid Discretizations",
)
