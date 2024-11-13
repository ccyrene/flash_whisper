from setuptools import setup, find_packages

with open('requirements.txt') as f:
    install_requires = f.read().splitlines()

setup(
    name="flash_whisper",
    version="1.0.0",
    author="rungrodks",
    author_email="rungrodks@hotmail.com",
    url="https://github.com/rungrodkspeed/flash_whisper",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=install_requires,
    python_requires=">=3.10",
)
