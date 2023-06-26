# LRMI
Code for the paper ''Boosting Document-Level Relation Extraction with Logical Rules' Mining and Injection", which is under EMNLP 2022 review.

For simplicity, we only supply the code with the strong backbone [ATLOP](https://arxiv.org/abs/2010.11304) being the backbone tested on the [DWIE](https://arxiv.org/abs/2009.12626) dataset. LRMI with other backbones and datasets are similar. This code is adapted from the repository of [ATLOP](https://github.com/wzhouad/ATLOP). Thanks for their excellent work. 

In addition, predictions used for analysis in our paper are provided. We also provide mined rules whose confidence is higher than the threshold `minC`.

## Requirements

* Python (tested on 3.6)

* apex==0.9.10dev

* cvxpy==1.1.18

* dill==0.3.4

* gurobipy==9.5.1 (Note that installation via pip may not work. Please request an evaluation license or a free academic license of Gurobi. More instructions can be found in [link](https://www.gurobi.com/free-trial/).)

* matplotlib==3.3.1

* numpy==1.19.2

* opt_einsum==3.3.0

* pandas==1.1.3

* scipy==1.2.0

* tqdm==4.50.0

* transformers==3.4.0

* ujson==4.0.2

* wandb==0.10.32

* torch==1.6.0

  We also exported `enviroment.yaml` and `requirements.txt`.

## Dataset
The training and development set of [DocRED](https://www.aclweb.org/anthology/P19-1074/) dataset can be downloaded at [link](https://github.com/thunlp/DocRED/tree/master/data). And the test set used in LRMI can be downloaded at [link](https://github.com/tonytan48/Re-DocRED).  The DWIE dataset can be obtained following the instructions in [LogiRE](https://github.com/rudongyu/LogiRE). We also upload the processed dataset in EMNLP 2022 START Conference Manager.

The expected structure of files is:

```
​```
ATLOP+LRMI
 |-- dataset_dwie
 |    |-- train_annotated.json        
 |    |-- dev.json
 |    |-- test.json
 |    |-- meta
 |    |    |-- ner2id.json        
 |    |    |-- rel2id.json
 |    |    |-- vec.npy
 |    |    |-- word2id.json
 |-- dataset_docred
 |    |-- train_annotated.json        
 |    |-- dev.json
 |    |-- test.json
 |    |-- rel_info.json
 |    |-- meta
 |    |    |-- ner2id.json        
 |    |    |-- rel2id.json
 |    |    |-- vec.npy
 |    |    |-- word2id.json
 |    |    |-- char_vec.npy
 |    |    |-- char2id.json
​```

```

## Pre-Trained Language Model 

Download `BERT-base-uncased` at [link](https://huggingface.co/bert-base-uncased/tree/main). And put downloaded files into `./PLM/bert-base-uncased` . The expected structure of files is:

```
​```
ATLOP+LRMI
 |-- PLM
 |    |-- bert-base-uncased
 |    |    |-- config.json        
 |    |    |-- pytorch_model.bin
 |    |    |-- vocab.txt
​```
```

## Mined Rules

We supply mined rules on DWIE and DocRED in `./mined_rules`. The structure of files is :

```
​```
ATLOP+LRMI
 |-- mined_rules:
 |    |-- rule_docred.txt
 |    |-- rule_dwie.txt
```

Examples are as follows:

`['in1', 'in0'] -> in0 : 1.0` means ` in0(h,t) ← in1(h,z) ⋀ in0(z,t)`,whose confidence is 1.0.   
`['anti_based_in2', 'based_in0'] -> in0 : 1.0 ` means  `in0(h,t) ← based_in2(z,h) ⋀ based_in0(z,t)`,whose confidence is 1.0.   

## Predictions Produced by Trained Models

We supply predictions produced by `ATLOP`, `ATLOP+LogiRE`, and `ATLOP+LRMI`. The structure of files is :

```
​```
ATLOP+LRMI
 |-- results_for_dwie
 |    |-- result_ATLOP_dev.json
 |    |-- result_ATLOP_test.json
 |    |-- result_LogiRE_test.json
 |    |-- result_LRMI_dev.json
 |    |-- result_LRMI_test.json
 |-- results_for_docred
 |    |-- result_ATLOP_test.json
 |    |-- result_LRMI_test.json

```

## Trained Models

We supply trained `ATLOP` & `ATLOP+LRMI` on the DWIE dataset in [link](https://drive.google.com/file/d/13u_BXjvpNl_3YpDtAQac_dWvfhpAw6Gp/view?usp=sharing) and [link](https://drive.google.com/file/d/1R7LE2rR_LHBoCEas62eZ3yriLtYa27MH/view?usp=sharing), respectively. Please download trained models and put them into the path `./trained_model/`. The expected structure of files is:

```
​```
ATLOP+LRMI
 |-- trained_model
 |    |-- model_ATLOP_DWIE.pth
 |    |-- model_LRMI_DWIE.pth

```

## Log Samples

 We also provide log samples in  `./logs/`. These samples involve the training and inference of `ATLOP+LRMI` and the inference of `ATLOP`. 

## Training and Evaluation of  ATLOP+LRMI

```bash
>> sh scripts/LRMI_train_DWIE.sh  # for training; if trained model has been downloaded, this process can be omitted
```

The classification loss, consistency regularization loss, total loss, and evaluation results on the dev set are synced to the wandb dashboard.
```bash
>> sh scripts/LRMI_evaluate_DWIE.sh  # for inference
```
The program will generate a test file `./results_for_dwie/result_LRMI.json` in the official evaluation format.  In addition, the log involving evaluation results would be dumped to `./logs/LRMI_DWIE_evaluation.out`.

Attention: There may be a bug when wandb is synchronized in the cloud. If this happens, try `wandb offline` in terminal. More instructions can be found in [link](https://docs.wandb.ai/ref/cli/wandb-offline) .

## Evaluation of  ATLOP

```bash
>> sh scripts/ATLOP_evaluate_DWIE.sh  # for inference 
```

The program will generate a test file `./results_for_dwie/result_ATLOP.json` in the official evaluation format.  In addition, the log involving evaluation results would be dumped to `./logs/ATLOP_DWIE_evaluation.out`.
