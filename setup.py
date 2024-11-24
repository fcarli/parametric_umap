from setuptools import setup, find_packages

setup(
    name="parametric_umap",
    version="0.1.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "torch",
        "faiss-cpu",  # or faiss-gpu
        "tqdm",
    ],
    author="Francesco Carli",
    author_email="francesco.carli94@gmail.com",
    description="A streamlined and fast implementation of parametric UMAP using PyTorch and FAISS",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/mr-fcharles/parametric_umap",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
