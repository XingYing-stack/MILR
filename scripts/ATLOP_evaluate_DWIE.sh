CUDA_VISIBLE_DEVICES=0 nohup python -u train.py --name ATLOP \
--dataset dwie \
--transformer_type bert \
--model_name_or_path ./PLM/bert-base-uncased \
--train_file train_annotated.json \
--dev_file dev.json \
--test_file test.json \
--load_path ./trained_model/model_ATLOP_DWIE.pth \
--num_train_epochs 30.0 \
--train_batch_size 4 \
--test_batch_size 4 \
--threshold_1_hop 0.1 \
--threshold_2_hop 0.005 \
--seed 66 \
--minC 0.98 \
--num_class 66 \
--tau 0.8 \
--k 0.5 \
--lambda_srr 0.001 > ./logs/ATLOP_DWIE_evaluation.out 2>&1 &