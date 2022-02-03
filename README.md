# cogrib - Cloud Optimized GRIB

This library has utilities for facilitating access to GRIB2 files stored in cloud Blob Storage.

## The problem

Traditionally, to load data from a GRIB2 file you would need to first download the entire GRIB2 file to disk and then read a portion of it.

1. Disks are relatively slow
2. GRIB files contain many variables, and you might just be interested in a small portion of it

We'd like to access subsets of the GRIB2 file directly from Blob Storage, without having to download all or part of the file locally first.

We *could* just convert the GRIB2 file to Zarr. But we'll assume that the data provider has to host the unmodified GRIB2 files and doesn't want to host two copies of the data.

## How it works

There are two distinct stages: First a data host scans the GRIB2 file for datasets and figures out which portions the GRIB2 file each dataset refers to.
These references, which are byte offsets and lengths for each variable, are saved off to a "kerchunk index file." The index file contains

1. All the metadata (the dimensions, coordinate values, each variable's attributes, etc.)
2. *References* to the GRIB2 as `(url, offset, length)` tuples.

In code, that looks like

```python
>>> datasets = cfgrib.open_datasets("/path/to/file.grib2")  # must be local
>>> references = [cogrib.make_references(ds) for ds in datasets]
```

These references would be provided as, e.g., assets on a STAC item or collection.

Second, a user loads up this kerchunk index file and loads it into xarray using the Zarr engine using the "normal" Kerchunk / fsspec reference access pattern. For those unfamiliar, with this process,
Zarr doesn't natively understand what to do with these references files. So we need a slightly intelligent Zarr store that will perform the HTTP range requests
for the actual data from the GRIB2 file. The fsspec "reference" filesystem does just that.

```python
>>> references = requests.get("http://path/to/references.json")
>>> store = fsspec.filesystem("reference", fo=references).get_mapper("")
>>> ds = xr.open_dataset(store, engine="zarr", chunks={})
```

## Why isn't this in kerchunk / fsspec's reference filesystem?

It probably should be. Just experimenting for now.