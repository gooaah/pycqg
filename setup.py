import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="pycqg",
    version="0.1",
    author="Hao Gao",
    author_email="gaaooh@126.com",
    description="A Python package for construction and analysis of crystal quotient graphs.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gooaah/pycqg",
    # include_package_data=True,
    # exclude_package_date={'':['.gitignore']},
    packages=setuptools.find_packages(),
    install_requires=[
        'numpy',
        'ase',
        'networkx'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
        "Operating System :: OS Independent",
    ],
)