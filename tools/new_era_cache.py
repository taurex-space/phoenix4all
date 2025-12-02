import json
import pathlib
import tarfile
from urllib.parse import urljoin

import fsspec
from indexed_gzip import IndexedGzipFile

BASE_URL = "https://www.fdr.uni-hamburg.de/record/17935/files/"

MODELS = [
    "PHOENIX-NewEraV3-GAIA-DR4_v3.4-SPECTRA",
    "PHOENIX-NewEraV3-JWST-SPECTRA",
    "PHOENIX-NewEraV3-LowRes-SPECTRA",
]

# Should be the project
output_directory = pathlib.Path("..") / "src" / "phoenix4all" / "cache" / "newera"

output_directory.mkdir(exist_ok=True)

for model in MODELS:
    output_directory_model = output_directory / model
    output_directory_model.mkdir(exist_ok=True)

    tar_index = []
    url = urljoin(BASE_URL, f"{model}.tar.gz?download=1")
    print(f"Processing {model} from {url}")
    with fsspec.open(url, mode="rb") as of:
        with IndexedGzipFile(fileobj=of, spacing=1024 * 1024 * 100) as gz:
            with tarfile.open(fileobj=gz, mode="r|") as tar:
                for member in tar:
                    print(member.name)
                    tar_index.append({
                        "name": member.name,
                        "offset_data": member.offset_data,
                        "offset": member.offset,
                        "size": member.size,
                    })
            gz.export_index(output_directory_model / f"{model}.gzidx")
        with open(output_directory_model / f"{model}_tar_index.json", "w") as f:
            json.dump(tar_index, f)
