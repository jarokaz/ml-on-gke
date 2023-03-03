# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import jax
import os
import socket
import time


from absl import flags
from absl import app
from absl import logging

from typing import Callable, Mapping, Optional, Sequence, Tuple, Type




if __name__ == '__main__':

  FLAGS = flags.FLAGS

  #jax.config.parse_flags_with_absl()

  flags.DEFINE_multi_string(
      'gin_file',
      default=None,
      help='Path to gin configuration file. Multiple paths may be passed and '
      'will be imported in the given order, with later configurations  '
      'overriding earlier ones.')

  flags.DEFINE_multi_string(
      'gin_bindings', default=[], help='Individual gin bindings.')

  flags.DEFINE_list(
      'gin_search_paths',
      default=['.'],
      help='Comma-separated list of gin config path prefixes to be prepended '
      'to suffixes given via `--gin_file`. If a file appears in. Only the '
      'first prefix that produces a valid path for each suffix will be '
      'used.')

  flags.DEFINE_string(
      'tfds_data_dir', None,
      'If set, this directory will be used to store datasets prepared by '
      'TensorFlow Datasets that are not available in the public TFDS GCS '
      'bucket. Note that this flag overrides the `tfds_data_dir` attribute of '
      'all `Task`s.')

  flags.DEFINE_list(
      'seqio_additional_cache_dirs', [],
      'Directories to search for cached Tasks in addition to defaults.')

  flags.DEFINE_boolean(
      'multiprocess_gpu',
      False,
      help='Initialize JAX distributed system for multi-host GPU, using '
      '`coordinator_address`, `process_count`, and `process_index`.')

  flags.DEFINE_string(
      'coordinator_address',
      None,
      help='IP address:port for multi-host GPU coordinator.')

  flags.DEFINE_integer(
      'process_count', None, help='Number of processes for multi-host GPU.')

  flags.DEFINE_integer('process_index', None, help='Index of this process.')


  def main(argv: Sequence[str]):
    """Wrapper for pdb post mortems."""
    _main(argv)

  def _main(argv: Sequence[str]):
      """True main function."""
      if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')
  
      if FLAGS.multiprocess_gpu and FLAGS.process_count > 1: 
          jax.distributed.initialize(coordinator_address=FLAGS.coordinator_address,
                                     num_processes=FLAGS.process_count,
                                     process_id=FLAGS.process_index)
  
      print(f'JAX global devices:{jax.devices()}')
      print(f'JAX local devices:{jax.local_devices()}')
  
      xs = jax.numpy.ones(jax.local_device_count())
      print(jax.pmap(lambda x: jax.lax.psum(x, 'i'), axis_name='i')(xs))
      print('Hooray ...')

  logging.info('Running training')
  app.run(_main)
   