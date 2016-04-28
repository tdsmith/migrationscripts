from setuptools import setup

versionfile = 'sting/version.py'
with open(versionfile, 'rb') as f:
    exec(compile(f.read(), versionfile, 'exec'))

setup(
    name='migration-sting',
    version=__version__,  # noqa
    url='https://github.com/tdsmith/migrationscripts',
    license='MIT',
    author='Tim D. Smith',
    author_email='tim.smith@uci.edu',
    description='Tools for migration tracking',
    packages=['sting'],
    platforms='any',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 2.7',
    ],
    install_requires=['numpy', 'pandas', 'Pillow', 'tifffile'],
    entry_points={
        'console_scripts': [
            'sting = sting.sting:main',
            'truncate_mdf = sting.truncate_mdf:main',
            'sting_time_series = sting.time_series:main',
            'tidycheck_mdf = sting.tidycheck_mdf:main',
            'reduce_mdf = sting.reduce_mdf:main',
            'sting_plot_displacement = sting.plot_displacement_per_interval:main',
            'sting_extrude = sting.extrude:main',
        ],
    },
)
