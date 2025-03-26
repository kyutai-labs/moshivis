# MoshiVis - MLX

See the [top-level README.md][main_repo] for more information on MoshiVis.

This is the MLX implementation for MoshiVis.

## Usage

We have tested the MLX version with MacBook Air M3 (4-bit quantization) and a MacMini M4 Pro (both 4- and 8-bit quantization).
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

```
@article{kyutai2025moshivis,
  author = {Amélie Royer and Moritz Böhle and Gabriel de Marmiesse and
  Laurent Mazaré and Alexandre Défossez and Neil Zeghidour and Patrick Pérez},
  year = {2025},
  title = {Vision-Speech Models: Teaching Speech Models to Converse about Images},
  journal = {ArXiv},
  url = {https://arxiv.org/abs/2503.15633}
}

@techreport{kyutai2024moshi,
      title={Moshi: a speech-text foundation model for real-time dialogue},
      author={Alexandre Défossez and Laurent Mazaré and Manu Orsini and
      Amélie Royer and Patrick Pérez and Hervé Jégou and Edouard Grave and Neil Zeghidour},
      year={2024},
      eprint={2410.00037},
      archivePrefix={arXiv},
      primaryClass={eess.AS},
      url={https://arxiv.org/abs/2410.00037},
}
```

[moshi]: https://kyutai.org/Moshi.pdf
[main_repo]: https://github.com/kyutai-labs/MoshiVis
