# SAMFinetune

SAMFinetune is a script for fine-tuning a pretrained SAM (Segment Anything Model) model on sidewalk data.

## Arguments

Here are the command-line arguments you can use with this script:

- `-m` or `--model`: The model to use for training. Can be either "base" (using model "facebook/sam-vit-base") or "huge" (using model "facebook/sam-vit-base"). The default is "base".

- `-b` or `--batch_size`: The batch size for training. The default is 2.

- `-e` or `--epochs`: The number of epochs for training. The default is 5.

- `-l` or `--learning_rate`: The learning rate for training. The default is 1e-5.

- `-c` or `--resume_training`: If this argument is included, training will resume from a checkpoint.

- `--checkpoint_path`: The path to save checkpoints. The default is "../models".

- `--data_path`: The path to save data. The default is "../data".

## Usage

To use this script, you would use a command like the following:

```bash
python SAMFinetune.py -m base -b 2 -e 5 -l 1e-5 -c --checkpoint_path ../models --data_path ../data
```

This command would start training with the base model, a batch size of 2, for 5 epochs, with a learning rate of 1e-5, resuming training from a checkpoint, and saving checkpoints and data to the default locations.