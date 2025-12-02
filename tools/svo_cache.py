import json
from dataclasses import asdict

from phoenix4all.io import json_zip
from phoenix4all.sources.svo import SVOSource

models = SVOSource.list_available_models()


data_set = {}


for model in models:
    print(f"Processing model: {model.id}")
    try:
        svo = SVOSource(model_name=model.id)
    except Exception as e:
        print(f"Failed to initialize SVOSource for model {model.id}: {e}")
        continue
    files = svo.list_available_files()
    data_set[model.id] = [asdict(f) for f in files]


with open("svo_dataset.json", "w") as f:
    json.dump(json_zip(data_set), f)
