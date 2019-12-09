import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="EasyDiff", # Replace with your own username
    version="0.0.2",
    author="CS207 Group18",
    author_email="kangliwu@hsph.harvard.edu",
    description="An auto differentiation library",
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