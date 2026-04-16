This repository contains a codebase to train and use an object detection model using the `transformers` library.

### Setup

Ensure that `uv` is present in the development environment, this can be done using `pip`.

Use python 3.11 (`uv python pin 3.11`).

As a fallback, `requirements.txt` is provided.  

### Training

Training data should be provided with the file structure:

```text
.
├── data
│   ├── test
│   │   └── metadata.jsonl
│   |   └── ...jpg
│   |   └── ...
│   ├── train
│   │   └── metadata.jsonl
│   |   └── ...jpg
│   |   └── ...
│   └── validation
│       └── metadata.jsonl
│       └── ...jpg
│       └── ...
├── src
│   ├── ...
|...
```

The structure of the `.jsonl` annotations is given in `example_metadata.jsonl` and the tools for writing to this format are provided in `src/tools/`

To train, simply `uv run train.py` with the necessary arguments. 

Training can be monitored on tensorboard:

`tensorboard --logdir=/absolute/path/to/runs --bind_all --port=<port>` 

### Inference

Use `infer.py` to run inference on a .mp4 video.

## Authors and acknowledgment

The training script builds on the work of the jupyter notebook that can be found [here](https://github.com/qubvel/transformers-notebooks/blob/main/notebooks/RT_DETR_v2_finetune_on_a_custom_dataset.ipynb).


## License
The content used in this project is under the Apache-2.0 license.
