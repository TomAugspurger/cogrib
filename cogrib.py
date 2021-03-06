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
>>> store = make_references(ds)
>>> my_indices = indices_for_dataset(ds)
>>> result = read(store, grib_url, my_indices)
>>> xr.testing.assert_array_equal(ds, result)
"""
from __future__ import annotations

import collections.abc
import itertools
import json
import datetime
import base64
import typing
import logging

import requests
import pandas as pd
import numcodecs
import xarray as xr
import cfgrib
import eccodes
import numpy as np

__version__ = "1.0.0"

logger = logging.getLogger(__name__)
# All of this logic for moving between index values / positions, Zarr
# keys, and xarray dimensions / coordinates is a mess. We're way
# too focused on this specific dataset and will need to generalize substantially.
# **Open problems**:
# 1. We're assuming all data variables are >=2d, using longitude & latitude as
#    the last two dimensions.
# 2. We're assuming some combination of "number" (ensemble member), `step` and
# "levelist" (isobaricInhPa) are the only "extra" dimensions.
# 3. ...
# Notes on "step"
# It just so happens that *one* of the files from the ECMWF uses `step` as


coordinate_name_to_index_key = {
    "isobaricInhPa": "levelist",
}


class IndexKey(typing.NamedTuple):
    param: str
    number: int | None
    step: datetime.timedelta | None
    levelist: float | None

    @classmethod
    def from_index(
        cls, index, include_step=False, include_number=False, include_levelist=False
    ):
        step = number = levelist = None
        if include_number and "number" in index:
            number = int(index["number"])
        if include_levelist and "levelist" in index:
            levelist = float(index["levelist"])
        # if "step" in index:
        if include_step:
            step = index["step"]  # might be a scalar or range
            if "-" in step and not step.startswith("-"):
                # Can we have negative steps?
                step = int(step.split("-")[1])
                # TODO: this assumes hours. That's not generally true.
            else:
                step = int(step)
            step = pd.Timedelta(hours=step)
        return cls(index["param"], number, step, levelist)


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


def make_references(ds: xr.DataArray, indices: list[Index], grib_url: str) -> dict:
    ds = prepare_write(ds)

    store = {}
    _ = ds.to_zarr(store, compute=False)
    keys_to_index = {
        IndexKey.from_index(
            v,
            include_number="number" in ds.dims,
            include_step="step" in ds.dims,
            include_levelist="isobaricInhPa" in ds.dims,
        ): v
        for v in indices
    }

    for var_name, v in ds.data_vars.items():
        variable_keys = index_keys_for_variable(v)
        dims = v.dims[:-2]
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
    result = x.attrs["GRIB_shortName"]
    if result == "tcwv":
        # Why? stream=oper, type=fc
        result = "tciwv"
    return result


def index_keys_for_variable(v: xr.DataArray) -> list[IndexKey]:
    short_name = [v.attrs["GRIB_shortName"]]
    step = number = levelist = [None]
    if "number" in v.dims:
        number = [int(i) for i in v.coords["number"].data.tolist()]
    if "step" in v.dims:
        step = pd.to_timedelta(v.coords["step"]).tolist()

    extra_dims = list(set(v.dims) - {"step", "number", "latitude", "longitude"})
    if extra_dims:
        assert len(extra_dims) == 1
        (dim,) = list(extra_dims)
        levelist = [float(i) for i in v.coords[dim].data.tolist()]
    return [IndexKey(*x) for x in itertools.product(short_name, number, step, levelist)]


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
        logger.info("GET %s - %s", grib_url, headers)
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


def merge(
    *references,
):
    ...


def name_dataset(ds: xr.DataArray) -> str:
    attrs = list(ds.data_vars.values())[0].attrs
    params = ["dataType", "typeOfLevel"]

    values = ["{}={}".format(k, attrs[f"GRIB_{k}"]) for k in params]
    k = attrs[f"GRIB_typeOfLevel"]
    v = ds[k]

    if v.size == 1:
        values.append("{}={}".format(k, v.item()))
    return "/".join(values)
