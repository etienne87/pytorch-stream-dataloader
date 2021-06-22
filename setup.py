from setuptools import setup, find_packages

setup(
  name = 'pytorch-streamloader',
  packages = find_packages(exclude=['legacy']),
  version = '0.19.6',
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
    'decord>=1.0',
    'skvideo>=1.1.11',
    'torch>=1.6',
    'torchvision'
  ],
  classifiers=[
    'Development Status :: 1 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3.6',
  ],
)
