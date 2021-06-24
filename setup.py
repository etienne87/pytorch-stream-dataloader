from setuptools import setup, find_packages

from pytorch_stream_dataloader import __version__


extra_video=[
    'decord',
    'scikit-video>=1.1.11',
    'torchvision>=0.9.0'
]

extra_test=[
    'pytest>=4',
    'pytest-cov>=2'
]

extra_dev=[
    *extra_test,
]

extra_ci = [
    *extra_test,
    'python-coveralls',
]


setup(
    name = 'pytorch-stream-dataloader',
    packages = find_packages(exclude=['legacy']),
    version = __version__,
    license='MIT',
    description = 'RNN DataLoader - Pytorch',
    author = 'Etienne Perot',
    author_email = 'et.perot@gmail.com',
    url = 'https://github.com/etienne87/pytorch-streamloader',
    keywords = [
        'artificial intelligence',
        'rnn data loading',
        'video/text/audio'
    ],
    install_requires=[
        'torch>=1.6',
    ],
    extras_require={
        'video':extra_video,
        'dev':extra_dev,
        'test':extra_test,
        'ci':extra_ci
    },
    classifiers=[
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
    ],
)
