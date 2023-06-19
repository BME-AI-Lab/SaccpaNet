from setuptools import find_packages
from setuptools import setup

setup(
    name="posture_experiment",
    version="0.1.0",
    install_requires=[],
    packages=find_packages("src"),
    package_dir={"": "src"},
    url="https://github.com/your-name/your_app",
    license="MIT",
    description="Shared common library for posture experiment",
)
