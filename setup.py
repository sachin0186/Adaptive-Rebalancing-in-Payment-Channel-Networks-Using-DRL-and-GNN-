from setuptools import setup, find_packages

setup(
    name="debal",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "simpy==4.1.1",
        "networkx==3.2.1",
        "torch==2.2.0",
        "torch-geometric==2.5.0",
        "gym>=0.17.0",
        "tensorboard>=2.4.0",
        "numpy>=1.19.0",
        "matplotlib>=3.3.0"
    ],
    python_requires=">=3.8",
) 