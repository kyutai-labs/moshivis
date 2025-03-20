# MoshiVis - PyTorch
TODO(amelie)
See the [top-level README.md][main_repo] for more information on MoshiVis. 
This is the PyTorch implementation for MoshiVis based on Moshi

[Moshi][moshi] is a speech-text foundation model and full-duplex spoken dialogue framework.
It uses [Mimi][moshi], a state-of-the-art streaming neural audio codec. Mimi operates at 12.5 Hz, and compresses
24 kHz audio down to 1.1 kbps, in a fully streaming manner (latency of 80ms, the frame size), yet performs better than existing, non-streaming, codec.



## Requirements

TODO(amelie)

## Usage

TODO(amelie)

Note: Padding and repeat peanlty from the UI npt configurable

## License

The present code is provided under the MIT license.


## Citation

If you use MoshiVis, please cite this repository and the Moshi paper.

```
@misc{kyutai2025moshivis,
  author = {},
  title = {MoshiVis: A trimodal model for natural conversations about images and more},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/kyutai-labs/moshi-vis}},
}

@techreport{kyutai2024moshi,
      title={Moshi: a speech-text foundation model for real-time dialogue},
      author={Alexandre D\'efossez and Laurent Mazar\'e and Manu Orsini and
      Am\'elie Royer and Patrick P\'erez and Herv\'e J\'egou and Edouard Grave and Neil Zeghidour},
      year={2024},
      eprint={2410.00037},
      archivePrefix={arXiv},
      primaryClass={eess.AS},
      url={https://arxiv.org/abs/2410.00037},
}
```


[main_repo]: https://github.com/kyutai-labs/moshivis