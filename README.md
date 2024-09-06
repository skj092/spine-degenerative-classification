# spine-degenerative-classification

## Wandb Log: https://wandb.ai/skj092/rsna24/table?nw=nwuserskj092



## Progress so far

1. Preproces Dicom files to PNG, and save them into folder categorised by class.
2. Preparing DataLoader for multi axial medical images.
3. Using Multi GPU for training.
4. Using Mixed Precision for training.
5. Using Wandb for logging.
6. Multi class classification loss handling.

## Ideas
1. Think of two step approach.
2. Utilize the co-ordinates data.


## To Read

- [ ] 1. https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/524500
- [x] 2. https://www.kaggle.com/competitions/rsna-2024-lumbar-spine-degenerative-classification/discussion/528653

## TODO
1. Make a submission with full training with current pipeline.
2. Pretrain the model on co-ordinates data, load this model for classification task. (To Read - 2)
