# MoshiVis - MLX

See the [top-level README.md][main_repo] for more information on MoshiVis.

This is the MLX implementation for MoshiVis.

## Usage

We have tested the MLX version with a Macbook Air M3 chip and a MacMini M4 Pro chip.
You can start the server with:

```bash
# In Bfloat16 - weights will be downloaded from HF
uv run server

# In Q4
uv run server -q 4

# In Q8
uv run server -q 8
```

This will start the web UI which you can connect to via http, at [localhost:8008](http://localhost:8008).

Note that unlike other backends, not all settings available in the web UI are propagated to the MLX backend. Instead, you can configure some options directly via the command line e.g. `--text-temperature`.

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
