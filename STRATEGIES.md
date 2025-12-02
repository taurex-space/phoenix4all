# Streaming tar.gz files with indexed_gzip

So the strategy is that we precompute the index_gzip which should allow for random access of the tar.gz file. Then we can use the tarfile module to stream the contents of the tar file.

We can precompute the index_gzip file with the following code:
```python
import indexed_gzip as igzip
import tarfile
import json

tarfilename = "path/to/your/phoenix/models.tar.gz"
gzfilename = tarfilename + ".gzidx"
tarindexfilename = tarfilename + ".taridx"
tar_index = []


myfile = igzip.IndexedGzipFile(tarfilename)
with tarfile.open(fileobj=myfile, mode="r|") as tar:
  for member in tar:
      tar_index.append({
          "name": member.name,
          "offset_data": member.offset_data,
          "offset": member.offset,
          "size": member.size
      })



import json

with open(tarindexfilename, 'w') as f:
    json.dump(tar_index, f)
```

Then we can use the following code to read a specific file from the tar.gz file:

```python
import indexed_gzip as igzip
import tarfile
import json

tarfilename = "path/to/your/phoenix/models.tar.gz"
indexfilename = tarfilename + ".gzidx"

## Replace with json load, just an example.
tar_index = [{'name': 'PHOENIX-NewEraV2-GAIA-DR4_v3.4-PHOTOMETRY.Z+0.5.txt',
  'offset_data': 512,
  'offset': 0,
  'size': 184224},
 {'name': 'PHOENIX-NewEraV2-GAIA-DR4_v3.4-PHOTOMETRY.Z-0.0.alpha=-0.2.txt',
  'offset_data': 185344,
  'offset': 184832,
  'size': 205678}]



myfile = igzip.IndexedGzipFile(tarfilename, index_file=indexfilename)
myfile.seek(tar_index[0]["offset"])
with tarfile.open(fileobj=myfile, mode="r|") as tar:
    member = tar.next()
    f = tar.extractfile(member).read()
print(f.decode('utf-8', errors='ignore'))
```

Why? We can then exploit the "Range" HTTP header to only download the bytes we need from a remote server!
Something like:

```python
import requests
headers = {"Range": f"bytes={start}-{end}"}
response = requests.get(url, headers=headers)
```


## Strategy

Precompute the indexed_gzip for common Phoenix databases (NewGen etc). Then use these and stream the specific bytes needed for a given model to unzip and read the file.

We can wrap an io class as a HTTPRangeReader or something that takes a URL and a byte range and returns a file-like object. Then we can pass this to indexed_gzip.IndexedGzipFile to create a random access gzip file over HTTP.

```python

import io
import requests
import indexed_gzip
import tarfile

class HTTPRangeReader(io.RawIOBase):
    """A minimal file-like object for reading remote files via HTTP range requests."""
    def __init__(self, url):
        self.url = url
        head = requests.head(url)
        if 'Content-Length' not in head.headers:
            raise RuntimeError("Server must provide Content-Length for range reads.")
        self.size = int(head.headers['Content-Length'])
        self.pos = 0

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        elif whence == io.SEEK_END:
            self.pos = self.size + offset
        return self.pos

    def tell(self):
        return self.pos

    def read(self, size=-1):
        if size < 0:
            end = self.size - 1
        else:
            end = min(self.pos + size - 1, self.size - 1)

        headers = {"Range": f"bytes={self.pos}-{end}"}
        r = requests.get(self.url, headers=headers)
        r.raise_for_status()
        data = r.content
        self.pos += len(data)
        return data
