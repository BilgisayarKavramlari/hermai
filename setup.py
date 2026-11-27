
import os
from setuptools import setup, find_packages

# README.md dosyasının yolunu güvenli bir şekilde al
here = os.path.abspath(os.path.dirname(__file__))
try:
    with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = 'A hermeneutic approach to explainable AI (XAI).'

setup(
    name='hermai',
    version='0.1.10',  
    author='sadi evren seker',
    author_email='hermai@sadievrenseker.com',
    description='A hermeneutic approach to explainable AI (XAI) with contextual and narrative explanations.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/bilgisayarkavramlari/hermai',
    project_urls={
        'Source': 'https://github.com/BilgisayarKavramlari/hermai',
        'Paper (HCT)': 'https://doi.org/10.3390/make8110381',
        'Verifier (xai-claims)': 'https://github.com/BilgisayarKavramlari/xai-claims',
    },
    packages=[
        'hermai',
        'hermai.core',
        'hermai.explainers',
        'hermai.perturbations',
        'hermai.visualizations'
    ],
    
    install_requires=[
        'pandas',
        'numpy',
        'scikit-learn',
        'matplotlib',
        'seaborn'
    ],
    include_package_data=True,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.8',
)