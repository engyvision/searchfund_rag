from setuptools import setup, find_packages

setup(
    name="webscraper",
    version="0.1",
    packages=find_packages(),
    include_package_data=True,
    # Make sure the src directory is treated as a package
    package_dir={"": "."},
    python_requires=">=3.7",
)