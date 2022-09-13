# GM-lncLoc
GM-lncLoc is a lncRNAs subcellular localization predictor based on Graph Neural Network with Meta-learning. On the one hand, it is based on the initial information extracted from the lncRNA sequence, and also combines the graph structure information to extract high level features of lncRNA. On the other hand, the training mode of meta-learning is introduced to obtain meta-parameters by training a series of tasks.

## Environmental requirements
```bash
torch == 1.5.0  
dgl == 0.4.3post2  
numpy  
psutil  
```

## Run
```bash
python train.py --data_dir data_dir/GM-lncLoc/ \  
                --epoch 5 \  
                --k_fold 1 \  
                --task_n 1100 \  
                --k_qry 10\  
                --k_spt 5 \  
                --update_lr 0.01 \  
                --update_step 5 \  
                --update_step_test 10 \  
                --meta_lr 5e-3 \  
                --num_workers 4 \  
                --train_result_report_steps 200 \  
                --hidden_dim 256 \  
                --task_num 4 \  
                --batchsz 500 \  
                --h 1  
```
## Pre-processing  
To prepare the input files(features.npy, graph_dgl.pkl and label.pkl):
```bash
1. k-mer: getting k-mer features
2. SMOTE: over-sampling
3. Constructing graph  
```
The codes are in the folder "data_process".

## Input files  
- `features.npy`: An array of arrays [feat_1, feat_2, ...] where feat_i is the feature matrix of graph i.
- `graph_dgl.pkl`: A list of DGL graph objects. For single graph G, use [G]; for multiple graph Gs, use [G1, G2, ...].
- `label.pkl`: A dictionary of labels where {'X_Y': Z} means the node Y in graph X has label Z.
- `train.csv`, `val.csv` and `test.csv`: Each file has two columns, the first one is 'X_Y' (node Y from graph X) and its label 'Z'.

## Output file  
- `result.txt`: including `hyper-parameters`, `Accurcy`, `Precision`, `Recall`, `F1`, `Sensitivity`, `Specificity`, `MCC`, `Max Momory` and `Time`.