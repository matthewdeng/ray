# LightGBMTrainer User Guide Summary

This document summarizes the comprehensive LightGBMTrainer user guide that was created following the same structure and style as the XGBoost user guide in [PR #52355](https://github.com/ray-project/ray/pull/52355).

## Files Created

### 1. LightGBM Quickstart Code (`doc/source/train/doc_code/lightgbm_quickstart.py`)

This file contains side-by-side code examples comparing:
- **Regular LightGBM training**: Basic LightGBM training using `lightgbm.train()` 
- **LightGBM + Ray Train**: Distributed training using `LightGBMTrainer`

Key features demonstrated:
- Loading data as Ray Data Datasets
- Converting dataset shards to `lightgbm.Dataset` objects
- Adding network parameters for distributed training with `ray.train.lightgbm.get_network_params()`
- Using `RayTrainReportCallback` for metrics and checkpointing
- Configuring scaling and launching distributed training
- Loading trained models from checkpoints

### 2. Main User Guide (`doc/source/train/getting-started-lightgbm.rst`)

A comprehensive 300+ line RST documentation file structured as follows:

#### Section Overview:
- **Quickstart**: High-level overview and side-by-side code comparison
- **Set up a training function**: How to wrap LightGBM code in a training function
- **Report metrics and save checkpoints**: Using `RayTrainReportCallback`
- **Loading data**: Techniques for distributed data loading
- **Use Ray Data to shard the dataset**: Automatic dataset sharding
- **Configure scale and GPUs**: Resource configuration and GPU support
- **Configure persistent storage**: Multi-node storage requirements
- **Launch a training job**: End-to-end training execution
- **Access training results**: Working with training results and checkpoints
- **Next steps**: Links to additional resources

#### Key Features:
- **LightGBM-specific parameters**: Uses appropriate LightGBM parameters like `objective: "regression"`, `metric: "l2"`, `boosting_type: "gbdt"`
- **GPU support**: Instructions for using `device: "gpu"` parameter
- **Network configuration**: Proper use of `ray.train.lightgbm.get_network_params()`
- **Model loading**: Correct usage of `lgb.Booster(model_file=model_path)` for checkpoint loading
- **Documentation links**: References to official LightGBM documentation

### 3. Navigation Update (`doc/source/train/train.rst`)

Updated the main Train documentation navigation to include:
```rst
LightGBM Guide <getting-started-lightgbm>
```

### 4. Notebook Deprecation Update

Updated the existing `distributed-xgboost-lightgbm.ipynb` notebook to include a reference to the new LightGBM guide in the deprecation notice:

```markdown
> **Note**: The API shown in this notebook is now deprecated. Please refer to the updated API in [Getting Started with Distributed Training using XGBoost](../../getting-started-xgboost.rst) and [Getting Started with Distributed Training using LightGBM](../../getting-started-lightgbm.rst) instead.
```

## Technical Details

### LightGBM-Specific Adaptations

The guide was adapted from the XGBoost version with the following LightGBM-specific changes:

1. **Parameter names**: 
   - XGBoost: `objective: "reg:squarederror"` → LightGBM: `objective: "regression"`
   - XGBoost: `tree_method: "approx"` → LightGBM: `boosting_type: "gbdt"`
   - XGBoost: `eta: 1e-4` → LightGBM: `learning_rate: 0.05`

2. **GPU configuration**:
   - XGBoost: `device: "cuda"` → LightGBM: `device: "gpu"`

3. **Dataset creation**:
   - Both use similar APIs: `lgb.Dataset(X, label=y)` vs `xgboost.DMatrix(X, label=y)`

4. **Model loading**:
   - XGBoost: `xgboost.Booster()` + `model.load_model()` → LightGBM: `lgb.Booster(model_file=path)`

5. **Network parameters**:
   - Uses `ray.train.lightgbm.get_network_params()` for distributed training setup

### Documentation Standards

The guide follows Ray documentation standards:
- **RST format** with proper Sphinx directives
- **Code examples** with `testcode` blocks (marked as `skipif: True` for examples)
- **Cross-references** to other Ray Train documentation
- **Warning boxes** for important notes
- **Tab-based comparisons** for before/after code
- **Proper API references** using Sphinx role syntax

## Integration with Existing Ray Train Documentation

The guide seamlessly integrates with the existing Ray Train documentation ecosystem:

1. **Follows established patterns** from PyTorch, PyTorch Lightning, and XGBoost guides
2. **Cross-references** other Ray Train concepts like ScalingConfig, RunConfig, DataConfig
3. **Links to Ray Data** documentation for dataset handling
4. **References fault tolerance** and Ray Tune integration capabilities
5. **Maintains consistency** with existing code examples and style

## Usage Instructions

Users can now:

1. **Navigate to the guide** from the main Ray Train documentation page
2. **Follow step-by-step instructions** to convert existing LightGBM code to use Ray Train
3. **Copy and adapt** the provided code examples for their specific use cases
4. **Scale their training** from single-machine to multi-node distributed training
5. **Leverage Ray ecosystem** features like fault tolerance and hyperparameter tuning

This comprehensive guide provides LightGBM users with the same level of documentation and support that was previously available only for other frameworks like PyTorch and XGBoost.