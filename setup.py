from setuptools import setup

DISTNAME = 'src'
DESCRIPTION = 'Baseband Digital Linear Modems'
LONG_DESCRIPTION = 'The documentation of this project can be obtained from our GitHub repository: https://github.com/kirlf/ModulationPy'
MAINTAINER = 'Vladimir Fadeev'
MAINTAINER_EMAIL = 'vovenur@gmail.com'
URL = 'https://github.com/kirlf/ModulationPy'
LICENSE = 'BSD 3-Clause'
VERSION = '0.1.5'


setup(
    name="ModulationPy",
    version=VERSION,
    
    # Project uses reStructuredText, so ensure that the docutils get
    # installed or upgraded on the target machine
        install_requires=[
          'numpy>=1.7.1',
          'matplotlib>=2.2.2',
    ],

    python_requires='>=3.6.4',
    #package_dir = {'': 'src'},
    packages = ['ModulationPy'],

    # metadata to display on PyPI
    license=LICENSE,
    author=MAINTAINER,
    author_email=MAINTAINER_EMAIL,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    keywords="communications digital modulation demodulation psk qam",
    url=URL,   # project home page, if any

    classifiers=[
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3.6',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Telecommunications Industry',
        'Operating System :: Unix',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development',
    ]

)
