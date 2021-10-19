from setuptools import setup

install_requires = [
    #numpy,
    #pandas,
    #sklearn,
]

packages = [
    'pandasxtend',
    'pandasxtend.catdap',
    'pandasxtend.eda',
]

console_scripts = [
    'pandasxtend_cli=pandasxtend_cli.call:main',
]


setup(
    name='pandasxtend',
    version='0.0.1',
    packages=packages,
    #install_requires=install_requires,
    entry_points={'console_scripts': console_scripts},
)