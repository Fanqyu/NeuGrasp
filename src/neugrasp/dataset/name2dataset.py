from ..dataset.train_dataset import GeneralRendererDataset, FinetuningRendererDataset, FinetuningRealDataset

name2dataset = {
    'gen': GeneralRendererDataset,
    'ft': FinetuningRendererDataset,
    'ft_real': FinetuningRealDataset
}
