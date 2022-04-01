from setuptools import setup, find_packages

with open("README.md", 'r') as readme_file:
	readme = readme_file.read()

with open("requirements.txt", 'r') as requirements_file:
	requirements = requirements_file.read().split("\n")

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