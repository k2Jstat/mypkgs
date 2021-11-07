from setuptools import setup

install_requires = [
    "numpy",
    "pandas",
    "seaborn",
    "joblib",
    "tqdm",
    "itertools",
    "matplotlib",
    "scipy"
    "sklearn",
]

packages = [
    'modelingset',
    'modelingset.lgbxtend',
    'modelingset.preprocessing',
]

console_scripts = [
    'modelingset_cli=modelingset_cli.call:main',
]


setup(
    name='modelingset',
    version='0.0.2',
    packages=packages,
    #install_requires=install_requires,
    entry_points={'console_scripts': console_scripts},
)