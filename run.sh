### train
python contrastive.py train --dataset solidletters --dataset_path /home/share/brep/abc/graph --max_epochs 300 --gpus -1 --random_rotate --batch_size 128 --wandb
python contrastive.py train --dataset solidletters --dataset_path /home/share/brep/abc/graph --max_epochs 100 --gpus -1 --random_rotate --batch_size 128 --wandb --channels "points,normals,tangents,trimming_mask"