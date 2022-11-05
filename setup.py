from setuptools import setup, find_packages

# pip install -e .
print(find_packages())
setup(
    name="myproject",
    version="0.1",
    packages=find_packages(),
)
