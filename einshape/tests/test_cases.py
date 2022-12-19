# coding=utf-8
# Copyright 2022 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Common test cases."""

TEST_CASES = [
    dict(
        testcase_name='simple_reshape',
        x=[3, 5],
        equation='i->i1',
        index_sizes={},
        expected=[[3], [5]]),
    dict(
        testcase_name='simple_transpose',
        x=[[7, 2, 4], [1, -3, 5]],
        equation='ij->ji',
        index_sizes={},
        expected=[[7, 1], [2, -3], [4, 5]]),
    dict(
        testcase_name='ungroup',
        x=[1, 4, 7, -2, 3, 2],
        equation='(ij)->ij',
        index_sizes=dict(j=3),
        expected=[[1, 4, 7], [-2, 3, 2]]),
    dict(
        testcase_name='tile_leading_dim',
        x=[3, 5],
        equation='j->nj',
        index_sizes=dict(n=3),
        expected=[[3, 5], [3, 5], [3, 5]]),
    dict(
        testcase_name='tile_trailing_dim',
        x=[3, 5],
        equation='j->jk',
        index_sizes=dict(k=3),
        expected=[[3, 3, 3], [5, 5, 5]]),
    dict(
        testcase_name='tile_multiple_dims',
        x=[3, 5],
        equation='j->njm',
        index_sizes=dict(n=3, m=4),
        expected=[[[3] * 4, [5] * 4], [[3] * 4, [5] * 4], [[3] * 4, [5] * 4]],
    )
]
