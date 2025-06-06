# Few-Shot Graph Out-of-Distribution Detection with LLMs

## Requirements
This code requires the following:

- Python==3.8
- PyTorch==2.0.1
- Numpy==1.24.4
- DGL==1.1.1+cu102
- torch_geometric

## Usage

below is to run the LLM-GOOD model
```
python ./GOOD/train_LLM_enhancer_reduce_cost.py
```

below is to run the LLM-GOOD-f model
```
python ./GOOD/train_LLM_enhancer.py
```

below is to run the baseline models
```
python ./GOOD/baseline.py
```

below is to run the query requests to LLM and save the results to a pt file in ./LLM/data/active 
the dataset can be modified by changing the "need_datasets" in the main function
```
python ./LLM/src/llm_zy_prompt.py
```