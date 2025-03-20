# MoshiVis - MLX

See the [top-level README.md][main_repo] for more information on MoshiVis.

This is the MLX implementation for MoshiVis.

## Usage

We have tested the MLX version with MacBook Pro M3.

We use (and recommend) `uv` to run the server. Start the server with:
```bash
uv run server
```

It starts the web UI. The connection is via http, at [localhost:8998](http://localhost:8998).

You can use `--hf-repo` to select a different pretrained model, by setting the proper Hugging Face repository.
See [the model list](https://github.com/kyutai-labs/moshi?tab=readme-ov-file#models) for a reference of the available models.

## License

The present code is provided under the MIT license.

Some of this code was taken from mlx-vlm v0.1.9, the code can be found here:
https://github.com/Blaizzy/mlx-vlm/tree/a11c034adf6ae4bca5a197990d1ecb77aba83c47

The license of mlx-vlm is MIT.


## Citation

If you use either Mimi or Moshi, please cite the following paper,

TODO: CHANGE ME
```
@techreport{kyutai2024moshi,
    author = {Alexandre D\'efossez and Laurent Mazar\'e and Manu Orsini and Am\'elie Royer and
			  Patrick P\'erez and Herv\'e J\'egou and Edouard Grave and Neil Zeghidour},
    title = {Moshi: a speech-text foundation model for real-time dialogue},
    institution = {Kyutai},
    year={2024},
    month={September},
    url={http://kyutai.org/Moshi.pdf},
}
```

[moshi]: https://kyutai.org/Moshi.pdf
[main_repo]: https://github.com/kyutai-labs/MoshiVis
