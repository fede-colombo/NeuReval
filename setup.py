import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="neureval",
    version="1.1.1",
    author="Federica Colombo",
    author_email="f.colombo8@studenti.unisr.it",
    description="Stability-based relative clustering validation algorithm for neuroimaging data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/fede-colombo/NeuReval",
    download_url="https://github.com/fede-colombo/NeuReval/releases/tag/v1.1.1",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=["numpy",
                      "scipy",
                      "scikit-learn",
                      "umap-learn",
                      "matplotlib"],
    python_requires='>=3.6',
)
