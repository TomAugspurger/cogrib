import docker
import pathlib
import pytest
import json
import xarray as xr
import azure.storage.blob
import fsspec
import cogrib
import cfgrib


HERE = pathlib.Path(__file__).parent
INDEX_FILE = HERE / "../data/20220126000000-0h-enfo-ef.index"
GRIB2_FILE = HERE / "../data/20220126000000-0h-enfo-ef.grib2"
GRIB2_PATH = "20220126/00z/0p4-beta/enfo/20220126000000-0h-enfo-ef.grib2"
GRIB2_URL = f"http://127.0.0.1:10000/devstoreaccount1/ecmwf/{GRIB2_PATH}"

URL = "http://127.0.0.1:10000"
ACCOUNT_NAME = "devstoreaccount1"
KEY = "Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw=="
CONN_STR = f"DefaultEndpointsProtocol=http;AccountName={ACCOUNT_NAME};AccountKey={KEY};BlobEndpoint={URL}/{ACCOUNT_NAME};"


with open(INDEX_FILE) as f:
    indices = [json.loads(x) for x in f.read().split("\n") if x]


@pytest.fixture(scope="session", autouse=True)
def spawn_azurite():
    client = docker.from_env()
    azurite = client.containers.run(
        "mcr.microsoft.com/azure-storage/azurite",
        "azurite-blob --loose --blobHost 0.0.0.0",
        detach=True,
        ports={"10000": "10000"},
        remove=True,
    )
    yield azurite
    azurite.stop()


@pytest.fixture(scope="module", autouse=True)
def storage():
    """
    Create blob using azurite.
    """
    conn_str = f"DefaultEndpointsProtocol=http;AccountName={ACCOUNT_NAME};AccountKey={KEY};BlobEndpoint={URL}/{ACCOUNT_NAME};"

    bsc = azure.storage.blob.BlobServiceClient.from_connection_string(conn_str=conn_str)
    bsc.create_container(
        "ecmwf", public_access=azure.storage.blob.PublicAccess.Container
    )
    container_client = bsc.get_container_client(container="ecmwf")
    with open(GRIB2_FILE, "rb") as f:
        container_client.upload_blob(GRIB2_PATH, f)

    sas = azure.storage.blob.generate_container_sas(
        ACCOUNT_NAME, "ecmwf", account_key=KEY
    )
    yield sas


datasets = cfgrib.open_datasets(str(GRIB2_FILE))


@pytest.mark.parametrize("ds", datasets)
def test_all(ds):
    sub_indices = cogrib.indices_for_dataset(ds, indices)
    store = cogrib.write(ds, sub_indices, GRIB2_URL)
    m = fsspec.filesystem("reference", fo=store).get_mapper("")
    result = xr.open_zarr(m).compute()
    xr.testing.assert_equal(result, ds)



# @pytest.fixture(scope="module")
# def ds_cf_single() -> xr.Dataset:
#     return xr.open_dataset(
#         str(GRIB2_FILE),
#         engine="cfgrib",
#         backend_kwargs={
#             "filter_by_keys": {"dataType": "cf", "typeOfLevel": "depthBelowLandLayer"}
#         },
#     )


# @pytest.fixture(scope="module")
# def ds_pf_single() -> xr.Dataset:
#     return xr.open_dataset(
#         str(GRIB2_FILE),
#         engine="cfgrib",
#         backend_kwargs={
#             "filter_by_keys": {"dataType": "pf", "typeOfLevel": "depthBelowLandLayer"}
#         },
#     )


# @pytest.fixture(scope="module")
# def ds_cf_multi() -> xr.Dataset:
#     ds = xr.open_dataset(
#         str(GRIB2_FILE),
#         engine="cfgrib",
#         backend_kwargs={
#             "filter_by_keys": {"dataType": "cf", "typeOfLevel": "isobaricInhPa"}
#         },
#     )
#     return ds


# @pytest.fixture(scope="module")
# def ds_pf_multi() -> xr.Dataset:
#     ds = xr.open_dataset(
#         str(GRIB2_FILE),
#         engine="cfgrib",
#         backend_kwargs={
#             "filter_by_keys": {"dataType": "pf", "typeOfLevel": "isobaricInhPa"}
#         },
#     )
#     return ds


# def test_cf_single(ds_cf_single):
#     ds = ds_cf_single

#     sub_indices = cogrib.indices_for_dataset(ds, indices)
#     store = cogrib.write(ds, sub_indices, GRIB2_URL)
#     m = fsspec.filesystem("reference", fo=store).get_mapper("")
#     result = xr.open_zarr(m).compute()
#     xr.testing.assert_equal(result, ds)


# def test_pf_single(ds_pf_single):
#     ds = ds_pf_single

#     sub_indices = cogrib.indices_for_dataset(ds, indices)
#     store = cogrib.write(ds, sub_indices, GRIB2_URL)
#     m = fsspec.filesystem("reference", fo=store).get_mapper("")
#     result = xr.open_zarr(m).compute()
#     xr.testing.assert_equal(result, ds)


# def test_cf_multi(ds_cf_multi):
#     ds = ds_cf_multi

#     sub_indices = cogrib.indices_for_dataset(ds, indices)
#     store = cogrib.write(ds, sub_indices, GRIB2_URL)
#     m = fsspec.filesystem("reference", fo=store).get_mapper("")
#     result = xr.open_zarr(m).compute()
#     xr.testing.assert_equal(result, ds)


# def test_pf_multi(ds_pf_multi):
#     ds = ds_pf_multi

#     sub_indices = cogrib.indices_for_dataset(ds, indices)
#     store = cogrib.write(ds, sub_indices, GRIB2_URL)
#     m = fsspec.filesystem("reference", fo=store).get_mapper("")
#     result = xr.open_zarr(m).compute()
#     xr.testing.assert_equal(result, ds)
