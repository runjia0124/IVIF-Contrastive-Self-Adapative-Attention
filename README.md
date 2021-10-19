# MACFuse:  Multi-level Attention-guided Contrastive Learning for Infrared and Visible Image Fusion
### Usage

#### Testing

`python main.py --test --use_gpu`

##### Training

1. Training stage: `python main.py --finetune_multiNegative --use_gpu --c1 0.75 --c2 0.5 --epoch 7 --bs 30`
2. Finetuning stage: `python main.py --train --use_gpu --c1 0.5 --c2 0.75 --contrast 1.0 --save_dir 0 --epoch 2 --bs 10`

Please check out `main.py` to find details on how to run this code.

