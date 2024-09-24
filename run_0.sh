### train
# python contrastive.py train --dataset solidletters --dataset_path /home/share/brep/abc/graph --max_epochs 300 --gpus -1 --random_rotate --batch_size 128 --wandb
# python contrastive.py train --dataset solidletters --dataset_path /home/share/brep/abc/graph --max_epochs 100 --gpus -1 --random_rotate --batch_size 128 --wandb --channels points,normals,tangents,trimming_mask

# python joint_predict.py traintest --gpus -1 --max_epochs 25 --wandb
# python joint_predict.py traintest --gpus -1 --max_epochs 25 --wandb --latent_dim 256 --n_head 4
# python joint_predict.py traintest --gpus -1 --max_epochs 25 --wandb --latent_dim 768 --n_head 12
# python joint_predict.py traintest --gpus -1 --max_epochs 25 --wandb --random_rotate
# python joint_predict.py traintest --gpus -1 --max_epochs 25 --wandb --lr 0.001
# python joint_predict.py traintest --gpus -1 --max_epochs 25 --wandb --n_layer_gat 0
# python joint_predict.py traintest --gpus -1 --max_epochs 25 --wandb --n_layer_sat 0
# python joint_predict.py traintest --gpus -1 --max_epochs 25 --wandb --n_layer_cat 0
# python joint_predict.py traintest --gpus -1 --max_epochs 25 --wandb --bias
# python joint_predict.py traintest --gpus -1 --max_epochs 25 --wandb --dropout 0.4
python joint_predict.py traintest --gpus -1 --max_epochs 25 --wandb --n_layer_gat 8
