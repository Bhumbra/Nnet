try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'Experimental deep learning framework',
    'author': 'Gary Bhumbra',
    'url': 'https://github.com/Bhumbra/Nnet',
    'author_email': 'bhumbra@gmail.com',
    'version': '0.0.1',
    'install_requires': ['nose'],
    'packages': ['nnet'],
    'scripts': [],
    'name': 'Nnet'
}

setup(**config)
