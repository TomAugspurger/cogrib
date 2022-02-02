"""
Cloud-optimized GRIB2 access.

GRIB2 files contain data that can form many xarray Datasets. To facilitate cloud-native access
over the network, this module provides

1. A utility for generating *small* Zarr-like stores where only the *metadata* are stored
   in the store. The actual (large) array values are stored in the native GRIB2 file.
2. A Zarr-compatible interface for loading those files on demand.

Examples
--------

>>> ds = xr.open_dataset(...)
>>> indices = [json.loads(x) for x in requests.get(...).text if x]
>>> store = write(ds)
>>> my_indices = indices_for_dataset(ds)
>>> result = read(store, grib_url, my_indices)
>>> xr.testing.assert_array_equal(ds, result)
"""
from __future__ import annotations

import collections.abc
import itertools
import json

import requests
import typing
import xarray as xr

import cfgrib
import eccodes
import numpy as np

__version__ = "1.0.0"


# All of this logic for moving between index values / positions, Zarr
# keys, and xarray dimensions / coordinates is a mess. We're way
# too focused on this specific dataset and will need to generalize substantially.
# **Open problems**:
# 1. We're assuming all data variables are >=2d, using longitude & latitude as
#    the last two dimensions.
# 2. We're assuming some combination of "number" (ensemble member) and "levelist"
#    (isobaricInhPa) are the only "extra" dimensions.
# 3. ...

# Design questions:
# 1. Where do we put the index information?
#    - Option 1: In the zarr store, as an array value.
#    - Option 2: In the zarr store, in `.zmetadata`.
#    - Option 3: Outside of the Zarr store (separate JSON file, STAC, etc.)
# 2. Who is responsible for translating the index metadata to an HTTP range request?
#    - Option 1: A custom Mapping
#    - Option 2: A custom Codec
# ...

coordinate_name_to_index_key = {
    "isobaricInhPa": "levelist",
}


class IndexKey(typing.NamedTuple):
    param: str
    number: int | None
    levelist: float | None

    @classmethod
    def from_index(cls, index):
        number = levelist = None
        if "number" in index:
            number = int(index["number"])
        if "levelist" in index:
            levelist = float(index["levelist"])
        return cls(index["param"], number, levelist)


Index = typing.TypedDict(
    "Index",
    {
        "_length": int,
        "_offset": int,
        "class": str,
        "date": str,
        "domain": str,
        "expver": str,
        "levelist": str,
        "levtype": str,
        "number": str,
        "param": str,
        "step": str,
        "stream": str,
        "time": str,
        "type": str,
    },
)


def write(ds: xr.DataArray, indices: list[Index]) -> dict:
    ds = prepare_write(ds)
    store: dict[str, bytes] = {}
    _ = ds.to_zarr(store, compute=False)
    keys_to_index = {IndexKey.from_index(v): v for v in indices}

    for var_name, v in ds.data_vars.items():
        dims = v.dims[:-2]  # TODO: generalize common dims
        ks = [[var_name]]

        if "number" in dims:
            ks.append(v.coords["number"].data)
        else:
            ks.append([None])

        for dim in dims:
            if dim in coordinate_name_to_index_key:
                ks.append(v.coords[dim].data)
        while len(ks) < len(IndexKey._fields):
            ks.append([None])

        variable_keys = list(itertools.product(*ks))
        # XXX: We assume the *order* of the dimensions matches the order in IndexKey
        # We know that can't actually be true...
        grib_indices = [keys_to_index[k] for k in variable_keys]
        zarr_indices = list(
            itertools.product(*[map(str, range(len(ds.coords[dim]))) for dim in dims])
        )
        base = "0.0"

        if len(dims) == 0:
            sep = ""
        else:
            sep = "."

        assert len(variable_keys) == len(zarr_indices)
        for zarr_index, grib_index in zip(zarr_indices, grib_indices):
            zidx = ".".join(zarr_index)
            zarr_key = f"{var_name}/{zidx}{sep}{base}"
            store[zarr_key] = json.dumps(grib_index).encode()

    return store


def read(store, grib_url, indices):
    mystore = COGRIBStore(store, grib_url, indices)
    return xr.open_zarr(mystore)


def index_key(x) -> IndexKey:
    return IndexKey(x["param"], x.get("number"), x.get("levelist"))


def index_variable_name(x):
    # some variables have different names in the index vs. Dataset.
    # TODO: see if cfgrib has a helper for this.
    if x == "tciwv":
        return "tcwv"
    elif x == "u10":
        return "10u"
    elif x == "v10":
        return "10v"
    elif x == "t2m":
        return "2t"
    else:
        return x


def indices_for_dataset(ds: xr.DataArray, indices: list[Index]) -> list[Index]:
    d = {index_key(idx): idx for idx in indices}
    keys = keys_from_dataset(ds)
    return [d[k] for k in keys]


def keys_from_dataset(ds: xr.Dataset) -> list[IndexKey]:
    """
    Discover the index keys for a particular dataset.
    """
    # this will need to be generalized much, much more.
    if set(ds.dims) == {"latitude", "longitude"}:
        return [IndexKey(index_variable_name(v), None, None) for v in ds.data_vars]
    elif set(ds.dims) == {"number", "latitude", "longitude"}:
        return [
            IndexKey(index_variable_name(v), str(i), None)
            for v in ds.data_vars
            for i in ds.number.data.tolist()
        ]
    elif set(ds.dims) == {"isobaricInhPa", "number", "latitude", "longitude"}:
        return [
            IndexKey(index_variable_name(v), str(i), str(int(p)))
            for v in ds.data_vars
            for i in ds.number.data.tolist()
            for p in ds.isobaricInhPa.data.tolist()
        ]
    elif set(ds.dims) == {"isobaricInhPa", "latitude", "longitude"}:
        return [
            IndexKey(index_variable_name(v), None, str(int(p)))
            for v in ds.data_vars
            for p in ds.isobaricInhPa.data.tolist()
        ]
    else:
        raise ValueError


def dataarray_from_index(grib_url: str, idxs: Index) -> np.ndarray:
    # TODO: think about batching HTTP requests.
    # Unclear how that fits in a regular __getitem__ API.
    # Might be doable with a getitems.
    # See if we can cut down on the number of HTTP requests.
    messages = []
    for idx in idxs:
        start_bytes = idx["_offset"]
        end_bytes = start_bytes + idx["_length"] - 1
        headers = dict(Range=f"bytes={start_bytes}-{end_bytes}")

        # TODO: sans-io
        r = requests.get(grib_url, headers=headers)
        r.raise_for_status()
        data = r.content
        h = eccodes.codes_new_from_message(data)
        messages.append(cfgrib.messages.Message(h))

    ds = xr.open_dataset(messages, engine="cfgrib")
    # Get just the data...
    return ds


def prepare_write(ds) -> xr.DataArray:
    ds = ds.copy()
    chunks = {}
    for k, v in ds.data_vars.items():
        for dim in v.dims[:-2]:
            chunks[dim] = 1
    ds = ds.chunk(chunks)

    for k, v in ds.data_vars.items():
        ds[k].encoding["compressor"] = None
    return ds


class COGRIBStore(collections.abc.Mapping):
    """
    A Zarr store for cloud-optimized GRIB file access.
    """

    def __init__(self, store, grib_url):
        self.store = store
        self.grib_url = grib_url

    def __getitem__(self, key) -> np.ndarray:
        result = self.store[key]
        # How can we avoid inferring whether or not the key is "special"?
        if ".zmetadata" in key or ".zarray" in key:
            return result
        elif isinstance(result, bytes) and result.startswith(b"{"):
            variable, _ = key.split("/")
            # ...
            index = json.loads(result)
            arr = dataarray_from_index(self.grib_url, [index])[variable]
            return arr.data
        else:
            return result

    def __iter__(self):
        yield from iter(self.store)

    def __len__(self):
        return len(self.store)

    # @property
    # def _group(self):
    #     return zarr.open(self.store)

    # @property
    # def _data_variables(self):
    #     return [k for k, v in self._group.items() if v.initialized == 0]

    # @property
    # def _extra_dim(self):
    #     k = self._data_variables[0]
    #     # TODO: generalize this.
    #     return self._group[k].ndim - 2

    # @property
    # def _extra_keys(self):
    #     """These are additional keys that aren't present in the Zarr store."""
    #     if self._extra_dim == 0:
    #         keys = set(
    #             f"{key}/{i}.0"
    #             for i in range(len(self.indices))
    #             for key in self._data_variables
    #         )
    #     else:
    #         keys = set()
    #         # this is wrong for multiple variables
    #         for key in self._data_variables:
    #             keys |= {f"{key}/{i}.0.0" for i in range(len(self.indices))}
    #     return keys


# What's missing: a mapping to / from variable/number to index positions


"""
Our mapping gets {variable}/{index-0}.{index-1}...0.0. So we need to know
how to go from that key to a specific grib index.

I think the best way is:
"""
