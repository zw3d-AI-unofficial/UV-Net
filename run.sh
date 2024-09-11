### train
python contrastive.py train --dataset solidletters --dataset_path /home/share/brep/abc/graph --max_epochs 300 --gpus -1 --random_rotate --batch_size 128 --wandb
python contrastive.py train --dataset solidletters --dataset_path /home/share/brep/abc/graph --max_epochs 100 --gpus -1 --random_rotate --batch_size 128 --wandb --channels points,normals,tangents,trimming_mask
python joint_predict.py traintest --dataset_path /home/share/brep/zw3d/joint/zw3d-joinable-dataset --gpus -1 --train_label_scheme Joint --max_epochs 50 --wandb
python joint_predict.py traintest --dataset_path /home/share/brep/zw3d/joint/zw3d-joinable-dataset --gpus -1 --train_label_scheme Joint --max_epochs 50 --wandb --lr 0.0001 --pretrained_model results/contrastive/0826/091553/best.ckpt
python joint_predict.py traintest --dataset_path /home/share/brep/zw3d/joint/zw3d-joinable-dataset --gpus -1 --train_label_scheme Joint --max_epochs 50 --wandb --channels points,normals,tangents,trimming_mask --lr 0.0001 --pretrained_model results/contrastive/0828/114150/best.ckpt

python joint_predict.py traintest --dataset /home/share/brep/zw3d/user-dataset/ --max_epoch 50 --gpus -1 --batch_size 16 --wandb