# Copyright 2021 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Setup for pip package."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest
from setuptools import find_packages
from setuptools import setup


REQUIRED_PACKAGES = [
    'absl-py',
    'dataclasses; python_version < "3.7"',
    'numpy',
]
EXTRA_PACKAGES = {
    'jax': ['jax>=0.1.71'],
    'jaxlib': ['jaxlib>=0.1.49'],
    'tensorflow': ['tensorflow>=1.8.0'],
    'tensorflow with gpu': ['tensorflow-gpu>=1.8.0'],
}


def einshape_test_suite():
  test_loader = unittest.TestLoader()
  test_suite = test_loader.discover('einshape/tests',
                                    pattern='*_test.py')
  return test_suite

setup(
    name='einshape',
    version='1.0',
    description='DSL-based reshaping library for JAX and other frameworks',
    url='https://github.com/deepmind/einshape',
    author='DeepMind',
    author_email='noreply@google.com',
    # Contained modules and scripts.
    packages=find_packages(),
    install_requires=REQUIRED_PACKAGES,
    extras_require=EXTRA_PACKAGES,
    platforms=['any'],
    license='Apache 2.0',
    test_suite='setup.einshape_test_suite',
)
