try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'name': 'cs282rl',
    'description': 'Domains and other useful code for Harvard CS 282r on Reinforcement Learning ',
    'author': 'Kenneth C. Arnold',
    'author_email': 'kcarnold@seas.harvard.edu',
    'version': '0.1',
    'packages': ['cs282rl'],
    'scripts': [],
}

setup(**config)
