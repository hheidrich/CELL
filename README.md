# CELL
This is the official repository of the ICML 2020 paper "NetGAN without GAN: From Random Walks to Low-Rank Approximations".

## Installation

The latest code can be installed directly from GitHub with:

```shell
$ pip install git+https://github.com/hheidrich/CELL.git
```

The code can be installed in development mode with:

```shell
$ git clone https://github.com/hheidrich/CELL.git
$ cd CELL
$ pip install -e .
```

Where `-e` means "editable" mode.

## Citation

@InProceedings{pmlr-v119-rendsburg20a,
  title = 	 {{N}et{GAN} without {GAN}: From Random Walks to Low-Rank Approximations},
  author =       {Rendsburg, Luca and Heidrich, Holger and Luxburg, Ulrike Von},
  booktitle = 	 {Proceedings of the 37th International Conference on Machine Learning},
  pages = 	 {8073--8082},
  year = 	 {2020},
  volume = 	 {119},
  series = 	 {Proceedings of Machine Learning Research},
  publisher =    {PMLR}
}

## References
### Cora dataset
Under `data/CORA-ML.npz` you can find the Cora-ML dataset. The raw data was originally published by   

McCallum, Andrew Kachites, Nigam, Kamal, Rennie, Jason, and Seymore, Kristie. *"Automating the construction of internet portals with machine learning."* Information Retrieval, 3(2):127–163, 2000.

and the graph was extracted by

Bojchevski, Aleksandar, and Stephan Günnemann. *"Deep gaussian embedding of attributed graphs: Unsupervised inductive learning via ranking."* ICLR 2018.

The files `data/CORA-ML_train.npz` and `link_prediction.p` contain the train-validation-test-split of `data/CORA-ML.npz` used in "NetGAN without GAN: From Random Walks to Low-Rank Approximations".
