# Compress the list of available files into


with open("hiresfit_cache.json") as f:
    result = json.load(f)


with open("hiresfit_cache.jsonz", "w") as f:
    json.dump(json_zip(result), f)
