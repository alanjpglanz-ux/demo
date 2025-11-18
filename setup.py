from setuptools import setup, find_packages

setup(
    name="autarkeia",
    version="0.1.0",
    packages=find_packages(),
    install_requires=["numpy", "matplotlib", "flask"],
    description="Autarkeia Metrics Checker - internal stability & emergence diagnostics",
    author="Alan J. Philip Glanz",
)
