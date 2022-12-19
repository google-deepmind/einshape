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

"""Tests for the TensorFlow einshape implementation."""

from absl.testing import absltest
from einshape.src.tensorflow import tf_ops
import tensorflow.compat.v1 as tf


class EinshapeTest(tf.test.TestCase):

  def test_simple_reshape(self):
    x = tf.constant([3, 5], dtype=tf.int64)
    y = tf_ops.einshape('i->i1', x)
    self.assertEqual([2, 1], y.shape.as_list())

    with tf.Session() as session:
      y_val = session.run(y)
    self.assertAllEqual([[3], [5]], y_val)

  def test_simple_transpose(self):
    x = tf.constant([[7, 2, 4], [1, -3, 5]], dtype=tf.int64)
    y = tf_ops.einshape('ij->ji', x)
    self.assertEqual([3, 2], y.shape.as_list())

    with tf.Session() as session:
      y_val = session.run(y)
    self.assertAllEqual([[7, 1], [2, -3], [4, 5]], y_val)

  def test_ungroup(self):
    x = tf.constant([1, 4, 7, -2, 3, 2], dtype=tf.int64)
    y = tf_ops.einshape('(ij)->ij', x, j=3)
    self.assertEqual([2, 3], y.shape.as_list())

    with tf.Session() as session:
      y_val = session.run(y)
    self.assertAllEqual([[1, 4, 7], [-2, 3, 2]], y_val)

  def test_reshape_unknown_input_shape(self):
    # Construct a tensor `x` whose shape is unknown at graph-time.
    n = tf.placeholder(shape=(), dtype=tf.int64)
    x = tf.constant([1, 4, 9])[:n]

    y = tf_ops.einshape('i->1i', x)
    self.assertEqual([1, None], y.shape.as_list())
    y_shape = tf.shape(y)

    with tf.Session() as session:
      y_val, y_shape_val = session.run((y, y_shape), feed_dict={n: 2})
    self.assertAllEqual([[1, 4]], y_val)
    self.assertAllEqual([1, 2], y_shape_val)

  def test_reshape_unknown_index_size(self):
    # Construct a shape hint whose value is unknown at graph-time.
    i = tf.placeholder(shape=(), dtype=tf.int64)
    x = tf.constant([1, 4, 9])

    y = tf_ops.einshape('i->1i', x, i=i)
    y_shape = tf.shape(y)

    with tf.Session() as session:
      y_val, y_shape_val = session.run((y, y_shape), feed_dict={i: 3})
    self.assertAllEqual([[1, 4, 9]], y_val)
    self.assertAllEqual([1, 3], y_shape_val)

  def test_transpose_unknown_input_shape(self):
    # Construct a tensor `x` whose shape is unknown at graph-time.
    n = tf.placeholder(shape=(), dtype=tf.int64)
    x = tf.constant([[1, 2], [4, -5], [9, 8], [16, 13]])[:n]

    y = tf_ops.einshape('ij->ji', x)
    self.assertEqual([2, None], y.shape.as_list())
    y_shape = tf.shape(y)

    with tf.Session() as session:
      y_val, y_shape_val = session.run((y, y_shape), feed_dict={n: 3})
    self.assertAllEqual([[1, 4, 9], [2, -5, 8]], y_val)
    self.assertAllEqual([2, 3], y_shape_val)

  def test_group_unknown_input_shape(self):
    # Construct a tensor `x` whose shape is unknown at graph-time.
    n = tf.placeholder(shape=(), dtype=tf.int64)
    x = tf.constant([[1, 2], [4, -5], [9, 8], [16, 13]])[:n]

    y = tf_ops.einshape('ij->(ij)', x)
    self.assertEqual([None], y.shape.as_list())
    y_shape = tf.shape(y)

    with tf.Session() as session:
      y_val, y_shape_val = session.run((y, y_shape), feed_dict={n: 3})
    self.assertAllEqual([1, 2, 4, -5, 9, 8], y_val)
    self.assertAllEqual([6], y_shape_val)

  def test_ungroup_unknown_input_shape(self):
    # Construct a tensor `x` whose shape is unknown at graph-time.
    n = tf.placeholder(shape=(), dtype=tf.int64)
    x = tf.constant([1, 2, 4, -5, 9, 8, 16, 13])[:n]

    y = tf_ops.einshape('(ij)->ij', x, j=2)
    self.assertEqual([None, 2], y.shape.as_list())
    y_shape = tf.shape(y)

    with tf.Session() as session:
      y_val, y_shape_val = session.run((y, y_shape), feed_dict={n: 6})
    self.assertAllEqual([[1, 2], [4, -5], [9, 8]], y_val)
    self.assertAllEqual([3, 2], y_shape_val)

  def test_ungroup_unknown_index_size(self):
    # Construct a shape hint whose value is unknown at graph-time.
    j = tf.placeholder(shape=(), dtype=tf.int64)
    x = tf.constant([1, 2, 4, -5, 9, 8])

    y = tf_ops.einshape('(ij)->ij', x, j=j)
    self.assertEqual([None, None], y.shape.as_list())
    y_shape = tf.shape(y)

    with tf.Session() as session:
      y_val, y_shape_val = session.run((y, y_shape), feed_dict={j: 3})
    self.assertAllEqual([[1, 2, 4], [-5, 9, 8]], y_val)
    self.assertAllEqual([2, 3], y_shape_val)

  def test_tile_leading_dim(self):
    x = tf.constant([3, 5], dtype=tf.int64)
    y = tf_ops.einshape('j->nj', x, n=3)
    self.assertEqual([3, 2], y.shape.as_list())

    with tf.Session() as session:
      y_val = session.run(y)
    self.assertAllEqual([[3, 5], [3, 5], [3, 5]], y_val)

  def test_tile_trailing_dim(self):
    x = tf.constant([3, 5], dtype=tf.int64)
    y = tf_ops.einshape('j->jk', x, k=3)
    self.assertEqual([2, 3], y.shape.as_list())

    with tf.Session() as session:
      y_val = session.run(y)
    self.assertAllEqual([[3, 3, 3], [5, 5, 5]], y_val)

  def test_accepts_python_list(self):
    x = [3, 5]  # Python list, not a TensorFlow tensor.
    y = tf_ops.einshape('i->i1', x)
    self.assertEqual([2, 1], y.shape.as_list())

    with tf.Session() as session:
      y_val = session.run(y)
    self.assertAllEqual([[3], [5]], y_val)


if __name__ == '__main__':
  tf.disable_v2_behavior()
  absltest.main()
