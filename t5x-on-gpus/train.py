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

flags.DEFINE_integer('num_processes', 1, 'Number of processes')
flags.DEFINE_string('job_name', None, 'Job name')
flags.DEFINE_string('sub_domain', None, 'Service sub domain')
flags.mark_flag_as_required('job_name')
flags.mark_flag_as_required('sub_domain')

FLAGS = flags.FLAGS

def _main(argv):

    job_completion_index = os.getenv("JOB_COMPLETION_INDEX")
    coordinator_fqdn = f'{FLAGS.job_name}-{job_completion_index}.{FLAGS.sub_domain}'
    print(f'Coordinator host name: {coordinator_fqdn}') 

    for retry_attempt in range(5):
        try:
            time.sleep(1)
            coordinator_ipaddress = socket.gethostbyname(coordinator_fqdn)
        except socket.gaierror:
            print(f'Failed to resolve: {coordinator_ipaddress}. Trying again in a second ...') 
        else:
            break

    print(f'Coordiantor IP address: {coordinator_ipaddress}')
    print(f'jax devices:{jax.devices()}')

if __name__ == "__main__":
    app.run(_main)
