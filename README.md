# CF-HyperGNNExplainer

# Installation and Setup
## 1. Prerequisites
- **uv:** This project uses `uv` for fast dependency management. Install it following the instructions at [https://docs.astral.sh/uv/](https://docs.astral.sh/uv/).
- **Python** >=3.14.

## 2. Install dependencies
Use `uv` to install the project's dependencies:

```bash
uv sync
```

## 3. Train the HGCN

```bash
cd src_sparse/
uv run train.py --dataset Cora --epochs 500
```

## 4. Generate the Counterfactual Explanations

```bash
uv run main_explain.py --dataset=cora --lr=0.1 --beta=0.5 --n-momentum=0.9 --cf-optimizer=SGD --strategy v1 --ckpt-path <path_to_pt_file>
```

## 5. Performance Evaluation

```bash
uv run evaluate.py --results <path_to_pkl_results_file> --strategy v1
```
