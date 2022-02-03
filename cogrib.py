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
import base64
import typing

import requests
import numcodecs
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


def write(ds: xr.DataArray, indices: list[Index], grib_url: str) -> dict:
    ds = prepare_write(ds)

    store = {}
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

    store = translate(store, grib_url)
    return store


def read(store, grib_url, indices):
    mystore = COGRIBStore(store, grib_url, indices)
    return xr.open_zarr(mystore)


def index_variable_name(x: xr.DataArray) -> str:
    return x.attrs["GRIB_shortName"]


def index_keys_for_variable(v: xr.DataArray) -> list[IndexKey]:
    short_name = [v.attrs["GRIB_shortName"]]
    number = levelist = [None]
    if "number" in v.dims:
        number = [int(i) for i in v.coords["number"].data.tolist()]
    extra_dims = list(set(v.dims) - {"number", "latitude", "longitude"})
    if extra_dims:
        assert len(extra_dims) == 1
        (dim,) = list(extra_dims)
        levelist = [float(i) for i in v.coords[dim].data.tolist()]
    return [IndexKey(*x) for x in itertools.product(short_name, number, levelist)]


def indices_for_dataset(ds: xr.DataArray, indices: list[Index]) -> list[Index]:
    d = {IndexKey.from_index(idx): idx for idx in indices}
    keys = index_keys_for_dataset(ds)
    return [d[k] for k in keys]


def index_keys_for_dataset(ds: xr.Dataset) -> list[IndexKey]:
    """
    Discover the index keys for a particular dataset.
    """
    keys = []
    for v in ds.data_vars.values():
        keys.extend(index_keys_for_variable(v))
    return keys


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

        # move http to fsspec ref
        print("GET", grib_url, headers)
        r = requests.get(grib_url, headers=headers)
        r.raise_for_status()
        # move this to a filter
        data = r.content
        h = eccodes.codes_new_from_message(data)
        messages.append(cfgrib.messages.Message(h))

    ds = xr.open_dataset(messages, engine="cfgrib")
    # Get just the data...
    return ds


class COGRIBFilter(numcodecs.abc.Codec):
    codec_id = "cogrib"

    def encode(self, buf):
        raise ValueError

    def decode(self, buf, out=None):
        h = eccodes.codes_new_from_message(buf)
        messages = [cfgrib.messages.Message(h)]
        ds = xr.open_dataset(messages, engine="cfgrib")

        result = ds[list(ds.data_vars)[0]].data

        if out is not None:
            out[:] = result
            result = out

        return result


numcodecs.register_codec(COGRIBFilter)


def prepare_write(ds) -> xr.DataArray:
    ds = ds.copy()
    chunks = {}
    filters = [COGRIBFilter()]
    for k, v in ds.data_vars.items():
        for dim in v.dims[:-2]:
            chunks[dim] = 1
    ds = ds.chunk(chunks)

    for k, v in ds.data_vars.items():
        ds[k].encoding["compressor"] = None
        ds[k].encoding["filters"] = filters
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


def translate(store: dict, grib_url: str) -> dict:
    """
    Translate our representation to a Kerchunk index file.
    """
    refs = {}
    for k, v in store.items():
        k2 = k.split("/")[-1]
        if k2 in {".zmetadata", ".zarray", ".zattrs", ".zgroup"}:
            refs[k] = v.decode()
        elif v.startswith(b"{"):
            index = json.loads(v)
            refs[k] = ["{{a}}", index["_offset"], index["_length"]]
        else:
            refs[k] = (b"base64:" + base64.b64encode(v)).decode()

    out = {
        "version": 1,
        "templates": {
            "a": grib_url,
        },
        "refs": refs,
    }
    return out
