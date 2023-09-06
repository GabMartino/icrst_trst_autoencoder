# Are we using Autoencoders in the wrong way?

Here we propose a different way to train a Classical Autoencoder.

The MNIST dataset class is modified sampling another observation from the same class distribution with a probability $p$.


```python

class RandomMNISTDataset(Dataset):
    def __init__(self, mnist_path, p=0.0, transform=None):
        super().__init__()
        self.train_ds = MNIST(
            mnist_path, train=True, download=True, transform=transform
        )
        self.p = p
        self.transform = None
        self.images = self.train_ds.data
        self.targets = self.train_ds.targets.numpy()
        self.distribution_dict = {}
        for idx, v in enumerate(self.targets):
            if v not in self.distribution_dict:
                self.distribution_dict[v] = [idx]
            else:
                self.distribution_dict[v].append(idx)

    def __len__(self):
        return self.targets.size

    def __getitem__(self, item):
        image, label = self.train_ds.__getitem__(item)  
        assert label == self.targets[item]
        from scipy.stats import bernoulli
        res = bernoulli.rvs(p=self.p)
        out = None
        if res == 1:
            idx = np.random.choice(self.distribution_dict[label])
            out, a_label = self.train_ds.__getitem__(idx)
        else:
            out = image

        return image, out, label
```
##References
```
@misc{martino2023using,
      title={Are We Using Autoencoders in a Wrong Way?}, 
      author={Gabriele Martino and Davide Moroni and Massimo Martinelli},
      year={2023},
      eprint={2309.01532},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
