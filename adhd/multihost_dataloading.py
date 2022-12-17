from collections import defaultdict  # pylint: disable=g-importing-member
from dataclasses import dataclass  # pylint: disable=g-importing-member
from functools import partial  # pylint: disable=g-importing-member
import os
from typing import Callable, Any, Dict, List, Tuple, Optional

import jax
from jax.experimental import global_device_array as gda_lib
from jax.experimental import PartitionSpec
# from jax.experimental.global_device_array import GlobalDeviceArray
from jax.experimental.maps import Mesh
from jax.experimental.pjit import pjit
from jax.experimental.pjit import with_sharding_constraint
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # disable CUDA not found etc warnings
import tensorflow as tf  # pylint: disable=g-import-not-at-top

# make pjit output GDAs
# jax.config.update('jax_parallel_functions_output_gda', True)

Pytree = Any
Device = Any

data_dim = 0  # assume data dimension is the first


# Per host data pipeline
# -----------------------------------------------------------------------------


def check_inputs(dataset, global_data_shape, data_axes):
  # TODO(sholto): Is there a way to do this without calling dataset?
  dataset_structure = jax.tree_util.tree_structure(iter(dataset).next())
  global_data_shape_structure = jax.tree_util.tree_structure(global_data_shape)
  data_axes_structure = jax.tree_util.tree_structure(data_axes)
  try:
    assert dataset_structure == global_data_shape_structure == data_axes_structure, 'All inputs should have the same pytree structure.'
  except AssertionError as msg:
    (print(
        f"""{msg} - The most likely reason for this is that global shapes should
        be arrays or claases not tuples - otherwise tree map enumerates indiviudal
        dimensions as leaves. Dataset: {dataset_structure}, \n Shapes:
          {global_data_shape_structure}, \n Axes: {data_axes_structure}"""))
  shapes, _ = jax.tree_util.tree_flatten(global_data_shape)
  batch_dims = [s[0] for s in shapes]
  assert all(b == batch_dims[0]for b in batch_dims), 'All batch axis should be equal for gdas'
  assert all(b[0] == shapes[0][0] for b in shapes), 'All dataset elements should be sharded along the data axis identically'
  batch_dim = batch_dims[0]
  print(f"{batch_dim=}")
  return batch_dim


def get_unique_shards(
    host_to_devices: Dict[int, List[Device]],
    device_to_index: Dict[Device, Tuple[slice, slice]]
) -> Tuple[Dict[int, int], Dict[int, int]]:
  """Looks at the sets of data each host needs, deduplicates, assigns a shard to the set."""

  host_to_dataset_shard = {}  # [process_id, index]
  dataset_shard_hash_to_index = {}  # [hash, index]

  for host_id, host_devices in host_to_devices.items():
    host_indices = [device_to_index[device] for device in host_devices]
    hashable_indices = jax.tree_map(lambda s: (s.start, s.stop), host_indices)
    pipeline_hash = hash(tuple(set(hashable_indices)))
    # assign each host's set of indices a shard index in the order we discover
    # this will be the shard index loaded by tf.data
    host_to_dataset_shard[host_id] = dataset_shard_hash_to_index.setdefault(
        pipeline_hash, len(dataset_shard_hash_to_index))

  # tf.data requires total num shards
  num_unique_shards = len(dataset_shard_hash_to_index)
  return host_to_dataset_shard, num_unique_shards


def convert_global_indices_to_local_indices(
    device_to_index: Dict[Device, Tuple[slice, slice]]
) -> Tuple[Dict[Device, slice], int]:
  """Converts global GDA indices for each device to local indices of host loaded data."""

  local_indices = [device_to_index[device] for device in jax.local_devices()]
  # Tacit assumption that we -only- shard dataset batch along data dim here, we could
  # relax this but I'm not sure it would actually be handled right by this approach:
  print(f"{local_indices=}")
  data_indices = [(s[data_dim].start, s[data_dim].stop) for s in local_indices]
  unique_slice_sizes = {idx: idx[1]-idx[0] for idx in data_indices}

  # assign a unique local data slice to each device
  total_data_to_load = 0
  device_index_hash_to_local_index = {}
  for idx, size in unique_slice_sizes.items():
    device_index_hash_to_local_index[idx] = slice(total_data_to_load, total_data_to_load + size)
    total_data_to_load += size

  device_to_local_indices = {}
  for device, data_index in zip(jax.local_devices(), data_indices):
    device_to_local_indices[device] = device_index_hash_to_local_index[data_index]

  return device_to_local_indices, total_data_to_load


def get_next_per_host(
    sharded_dataset: tf.data.Dataset,
    host_local_indices: Dict[Device, slice],
    global_data_shape: Pytree,
    global_mesh: Mesh,
    data_axes: PartitionSpec
) -> jax.Array:
  """Get device buffers to form GDA using per host pipeline."""

  # load from a single pipeline for the entire host
  # this is returned as a pytree in the same shape as global data shape

  # Slice this up using local indices and give it to the host local devices
  def form_gda(element, shape, axes) -> jax.Array:
    device_buffers = []
    for device in jax.local_devices():
      local_indices = host_local_indices[device]
      data = element[local_indices]
      device_buffers.append(jax.device_put(data, device))
    # return GlobalDeviceArray(shape, global_mesh, axes, device_buffers)
    return jax.make_array_from_single_device_arrays(
        shape, jax.sharding.MeshPspecSharding(global_mesh, axes), device_buffers)

  # local_data = sharded_dataset.next()
  #  pytree_of_gdas = jax.tree_map(
  #      form_gda, local_data, global_data_shape, data_axes)
  #  return pytree_of_gdas

  return (jax.tree_map(form_gda, x, global_data_shape, data_axes) for x in sharded_dataset)



def get_per_host_data_pipeline(
    dataset: tf.data.Dataset,
    global_data_shape: np.ndarray,
    global_mesh: Mesh,
    data_axes: PartitionSpec,
    batching_fn: Optional[Callable] = None,
) -> Callable[[], Pytree]:
  """Test the case where we have one data pipeline per host.

  To do this, we determine which pieces of data each host needs to feed it's
  devices, identify the unique sets of these (which is likely < num_hosts),
  and then create a data pipeline for each set.

  Args:
    dataset: tf dataset over all files
    global_data_shape: what the size of the GDA should be
    global_mesh: global devices mesh
    data_axes: axes along which data is partitioned

  Returns:
    sharded_dataset: Correct dataset to load for this host
    host_local_indices: indices for just the data loaded by the host's pipeline
  """

  check_inputs(dataset, global_data_shape, data_axes)

  # pytree of 'device_to_index' objects matching the structure of data
  device_to_index = jax.tree_map(
    lambda shape, axes: gda_lib.get_shard_indices(shape, global_mesh, axes),
    global_data_shape,
    data_axes)

  print(f"{device_to_index=}")

  # group by host_id
  host_to_devices = defaultdict(list)
  for d in jax.devices():
    host_to_devices[d.host_id].append(d)

  # Now, we want to find the number of unique (per host) dataset shards which
  # should be loaded and assign each host to their shard.

  # Now, as we are creating our own slice in this function, and assuming that
  # we only have one dimension we are sharding along, we don't need to do
  # clever tree mapping as the unique shards -> therefore just take
  # the first one and get the unique sharding from that.
  dataset_structure = jax.tree_util.tree_structure(global_data_shape)
  representative_device_to_index = dataset_structure.flatten_up_to(
      device_to_index)[0]
  host_to_dataset_shard, num_shards = get_unique_shards(
    host_to_devices, representative_device_to_index)
  # And assign devices indices into the data to be loaded by the host
  # The slices generated here are only along the batch dim, and thus will work
  # for all items in the data output pytree
  host_local_indices, total_data_to_load = convert_global_indices_to_local_indices(
      representative_device_to_index)

  # Create the data pipeline
  local_data_shard_index = host_to_dataset_shard[jax.process_index()]
  dataset = dataset.shard(num_shards=num_shards, index=local_data_shard_index)

  if batching_fn:
      dataset = batching_fn(dataset, total_data_to_load)
  else:
      dataset = dataset.batch(total_data_to_load).repeat()

  sharded_dataset = iter(dataset.as_numpy_iterator())

  return get_next_per_host(
      sharded_dataset,
      host_local_indices,
      global_data_shape,
      global_mesh,
      data_axes
  )

#   next_fn = partial(
#       get_next_per_host,
#       sharded_dataset,
#       host_local_indices,
#       global_data_shape,
#       global_mesh,
#       data_axes,
#   )
#   return next_fn