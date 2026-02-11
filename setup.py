from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="fraud-detection-topology",
    version="0.1.0",
    author="Matías Sánchez",
    description="Financial fraud detection using topology and directed graphs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mattsa-99/tesis-maestria",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scipy>=1.10.0",
        "networkx>=3.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "jupyterlab>=4.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.3.0",
            "pytest-cov>=4.1.0",
            "black>=23.3.0",
            "flake8>=6.0.0",
        ],
        "topology": [
            "gudhi>=3.8.0",
            "giotto-tda>=0.6.0",
        ],
        "visualization": [
            "plotly>=5.14.0",
            "pyvis>=0.3.0",
        ],
    },
)
