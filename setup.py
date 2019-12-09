import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="easydiff",
    version="0.0.3",
    author="Yang Zhou, Ruby Zhang, Kangli Wu, and Emily Gould",
    author_email="yangzhou@g.harvard.edu, yiqingzhang@fas.harvard.edu, kangliwu@hsph.harvard.edu, egould@mba2020.hbs.edu",
    description="An automatic differentiation library (support forward and reverse mode)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/CrackAD/cs207-FinalProject",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)