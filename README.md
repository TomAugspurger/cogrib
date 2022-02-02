# cogrib - Cloud Optimized GRIB

This library has utilities for facilitating access to GRIB2 files stored in cloud Blob Storage.

## The problem

Traditionally, to load data from a GRIB2 file you would need to first download the entire GRIB2 file to disk and then read a portion of it.

1. Disks are relatively slow
2. GRIB files contain many variables, and you might just be interested in a small portion of it

We'd like to access subsets of the GRIB2 file directly from Blob Storage, without having to download all or part of the file locally first.

We *could* just convert the GRIB2 file to Zarr. But we'll assume that the data provider has to host the unmodified GRIB2 files and doesn't want to host two copies of the data.

## How it works

From the user's point of view, access to the data should feel a lot like using Zarr:

```python
>>> store = cogrib.COGRIBStore(..., grib_url="https://path/to/original/grib/file.grib2")
>>> ds = xr.open_dataset(store, engine="zarr", chunks={})
```

The `store` contains all the *metadata* for the file: things like the dimension names, coordinate labels, attributes, etc. Crucially, the data variables are not
in `store`, since we don't want to host two copies of the data. Instead, we include "index lines", like the following:

```json
{"domain": "g", "date": "20220126", "time": "0000", "expver": "0001", "class": "od", "type": "pf", "stream": "enfo", "step": "306", "levtype": "sfc", "number": "18", "param": "10u", "_offset": 0, "_length": 609069}
```

These index lines, provided by the upstream data provider like the ECMWF, contain all the information necessary to fetch a chunk of data out of the original GRIB2 file
from an HTTP server that supports range requests.

When a user asks for some actual data, via something like `ds[variable].plot()`, the `cogrib.COGRIBStore` mapping makes the HTTP request for those bytes. From
the point of view of xarray, it's identical to any other Zarr store.

## Why isn't this in kerchunk / fsspec's reference filesystem?

It probably should be.