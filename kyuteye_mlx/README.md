# MoshiVis - MLX

See the [top-level README.md][main_repo] for more information on MoshiVis.

This is the MLX implementation for MoshiVis.

## Usage

We have tested the MLX version with MacBook Air M3 (4-bit quantization) and a MacMini M4 Pro (both 4- and 8-bit quantization).

We use (and recommend) `uv` to run the server. Start the server with:
```bash
uv run server
```

It starts the web UI. The connection is via http, at [localhost:8998](http://localhost:8998).

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
