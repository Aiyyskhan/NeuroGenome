from gettext import install
from importlib_metadata import version
from setuptools import setup, find_packages

with open("README.md", 'r') as readme_file:
	readme = readme_file.read()

requirements = ["numpy>=1.18.0", "h5py==2.10"]

setup(
	name="neurogenome",
	version="0.1.0",
	author="Aiyyskhan Alekseev",
	author_email="aiyyskhan@gmail.com",
	url="https://github.com/Aiyyskhan/NeuroGenome",
	license="MIT",
	description="A universal API for create artificial neural networks with a genetic code",
	long_description=readme,
	packages=find_packages(),
	install_requires=requirements,
	classifiers=[
		"Development Status :: 2 - Pre-Alpha",
		"Programming Language :: Python :: 3.7",
		"Programming Language :: Python :: 3.8",
		"Programming Language :: Python :: 3.9",
		"License :: OSI Approved :: MIT License"
	]
)