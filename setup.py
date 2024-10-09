"""Setup.py for pysentence-similarity package."""
from setuptools import setup, find_packages

setup(
    name="pysentence-similarity",
    version="1.0.1",
    author="goldpulpy",
    author_email="sawsani1928@gmail.com",
    description="pysentence-similarity this tool is designed to identify "
    "and find similarities between sentences and the base sentence "
    "in python language",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/goldpulpy/pysentence-similarity",
    project_urls={
        "Documentation": "https://github.com/goldpulpy/"
        "pysentence-similarity#readme",
        "Issues": "https://github.com/goldpulpy/"
        "pysentence-similarity/issues",
        "Homepage": "https://github.com/goldpulpy/pysentence-similarity",
    },
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Natural Language :: English",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Intended Audience :: Developers",
    ],
    keywords="sentence similarity NLP natural language processing",
    python_requires='>=3.8',
    install_requires=[
        "tokenizers",
        "onnxruntime-gpu",
        "beautifulsoup4",
        "platformdirs",
        "h5py"
    ],
)
