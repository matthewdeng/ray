# LightGBM Examples with Ray Train

This directory contains examples and guides for using LightGBM with Ray Train for distributed training.

## Overview

Ray Train provides seamless integration with LightGBM for distributed gradient boosting. Key features include:

- **Distributed Training**: Scale LightGBM training across multiple workers
- **Fault Tolerance**: Automatic recovery from worker failures
- **Data Integration**: Seamless integration with Ray Data for preprocessing
- **Checkpointing**: Built-in model checkpointing and metrics reporting
- **Resource Management**: Efficient allocation of CPU and GPU resources

## Quick Start

Here's a minimal example to get you started:

```python
import ray
from ray.train.lightgbm import LightGBMTrainer, RayTrainReportCallback
from ray.train import ScalingConfig
import lightgbm as lgb

# Load data
train_dataset = ray.data.read_csv("path/to/train.csv")
valid_dataset = ray.data.read_csv("path/to/valid.csv")

def train_func():
    # Get data shards
    train_shard = ray.train.get_dataset_shard("train")
    valid_shard = ray.train.get_dataset_shard("valid")
    
    # Convert to pandas
    train_df = train_shard.to_pandas()
    valid_df = valid_shard.to_pandas()
    
    # Prepare data
    train_X, train_y = train_df.drop("target", axis=1), train_df["target"]
    valid_X, valid_y = valid_df.drop("target", axis=1), valid_df["target"]
    
    train_set = lgb.Dataset(train_X, label=train_y)
    valid_set = lgb.Dataset(valid_X, label=valid_y)
    
    # Train model
    model = lgb.train(
        {"objective": "binary", "metric": "binary_logloss"},
        train_set,
        valid_sets=[valid_set],
        num_boost_round=100,
        callbacks=[RayTrainReportCallback()]
    )

# Create trainer
trainer = LightGBMTrainer(
    train_func,
    scaling_config=ScalingConfig(num_workers=2),
    datasets={"train": train_dataset, "valid": valid_dataset}
)

# Start training
result = trainer.fit()
```

## Examples

### 1. Basic Binary Classification

```python
import ray
from ray.train.lightgbm import LightGBMTrainer, RayTrainReportCallback
from ray.train import ScalingConfig, RunConfig, CheckpointConfig
import lightgbm as lgb

def binary_classification_train():
    # Load breast cancer dataset
    dataset = ray.data.read_csv("s3://anonymous@air-example-data/breast_cancer.csv")
    train_ds, valid_ds = dataset.train_test_split(test_size=0.2)
    
    def train_func():
        # Get data shards
        train_shard = ray.train.get_dataset_shard("train")
        valid_shard = ray.train.get_dataset_shard("valid")
        
        train_df = train_shard.to_pandas()
        valid_df = valid_shard.to_pandas()
        
        # Prepare features and labels
        train_X, train_y = train_df.drop("target", axis=1), train_df["target"]
        valid_X, valid_y = valid_df.drop("target", axis=1), valid_df["target"]
        
        train_set = lgb.Dataset(train_X, label=train_y)
        valid_set = lgb.Dataset(valid_X, label=valid_y)
        
        # LightGBM parameters
        params = {
            "objective": "binary",
            "metric": ["binary_logloss", "binary_error"],
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "verbose": -1,
        }
        
        # Train with Ray Train callback
        model = lgb.train(
            params,
            train_set,
            valid_sets=[valid_set],
            valid_names=["eval"],
            num_boost_round=100,
            callbacks=[RayTrainReportCallback()]
        )
    
    # Configure training
    trainer = LightGBMTrainer(
        train_func,
        scaling_config=ScalingConfig(num_workers=2, resources_per_worker={"CPU": 2}),
        run_config=RunConfig(
            checkpoint_config=CheckpointConfig(checkpoint_frequency=10, num_to_keep=2)
        ),
        datasets={"train": train_ds, "valid": valid_ds}
    )
    
    return trainer.fit()

# Run training
result = binary_classification_train()
print(f"Training completed! Best metrics: {result.metrics}")
```

### 2. Multi-class Classification

```python
def multiclass_classification_train():
    # Load iris dataset
    dataset = ray.data.read_csv("s3://anonymous@air-example-data/iris.csv")
    train_ds, valid_ds = dataset.train_test_split(test_size=0.2)
    
    def train_func():
        train_shard = ray.train.get_dataset_shard("train")
        valid_shard = ray.train.get_dataset_shard("valid")
        
        train_df = train_shard.to_pandas()
        valid_df = valid_shard.to_pandas()
        
        train_X, train_y = train_df.drop("target", axis=1), train_df["target"]
        valid_X, valid_y = valid_df.drop("target", axis=1), valid_df["target"]
        
        train_set = lgb.Dataset(train_X, label=train_y)
        valid_set = lgb.Dataset(valid_X, label=valid_y)
        
        params = {
            "objective": "multiclass",
            "num_class": 3,
            "metric": "multi_logloss",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "verbose": -1,
        }
        
        model = lgb.train(
            params,
            train_set,
            valid_sets=[valid_set],
            num_boost_round=100,
            callbacks=[RayTrainReportCallback()]
        )
    
    trainer = LightGBMTrainer(
        train_func,
        scaling_config=ScalingConfig(num_workers=2),
        datasets={"train": train_ds, "valid": valid_ds}
    )
    
    return trainer.fit()
```

### 3. Regression

```python
def regression_train():
    # Load housing dataset
    dataset = ray.data.read_csv("s3://anonymous@air-example-data/house_prices.csv")
    train_ds, valid_ds = dataset.train_test_split(test_size=0.2)
    
    def train_func():
        train_shard = ray.train.get_dataset_shard("train")
        valid_shard = ray.train.get_dataset_shard("valid")
        
        train_df = train_shard.to_pandas()
        valid_df = valid_shard.to_pandas()
        
        train_X, train_y = train_df.drop("price", axis=1), train_df["price"]
        valid_X, valid_y = valid_df.drop("price", axis=1), valid_df["price"]
        
        train_set = lgb.Dataset(train_X, label=train_y)
        valid_set = lgb.Dataset(valid_X, label=valid_y)
        
        params = {
            "objective": "regression",
            "metric": "rmse",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "verbose": -1,
        }
        
        model = lgb.train(
            params,
            train_set,
            valid_sets=[valid_set],
            num_boost_round=100,
            callbacks=[RayTrainReportCallback()]
        )
    
    trainer = LightGBMTrainer(
        train_func,
        scaling_config=ScalingConfig(num_workers=2),
        datasets={"train": train_ds, "valid": valid_ds}
    )
    
    return trainer.fit()
```

### 4. GPU Training

```python
def gpu_training():
    dataset = ray.data.read_csv("s3://anonymous@air-example-data/breast_cancer.csv")
    train_ds, valid_ds = dataset.train_test_split(test_size=0.2)
    
    def train_func():
        train_shard = ray.train.get_dataset_shard("train")
        valid_shard = ray.train.get_dataset_shard("valid")
        
        train_df = train_shard.to_pandas()
        valid_df = valid_shard.to_pandas()
        
        train_X, train_y = train_df.drop("target", axis=1), train_df["target"]
        valid_X, valid_y = valid_df.drop("target", axis=1), valid_df["target"]
        
        train_set = lgb.Dataset(train_X, label=train_y)
        valid_set = lgb.Dataset(valid_X, label=valid_y)
        
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "device": "gpu",  # Enable GPU training
            "gpu_platform_id": 0,
            "gpu_device_id": 0,
            "verbose": -1,
        }
        
        model = lgb.train(
            params,
            train_set,
            valid_sets=[valid_set],
            num_boost_round=100,
            callbacks=[RayTrainReportCallback()]
        )
    
    trainer = LightGBMTrainer(
        train_func,
        scaling_config=ScalingConfig(num_workers=2, use_gpu=True),
        datasets={"train": train_ds, "valid": valid_ds}
    )
    
    return trainer.fit()
```

### 5. Advanced Data Preprocessing

```python
from ray.data.preprocessors import StandardScaler, OneHotEncoder

def advanced_preprocessing():
    dataset = ray.data.read_csv("s3://anonymous@air-example-data/adult_census.csv")
    
    # Apply preprocessing
    preprocessor = StandardScaler(columns=["age", "hours_per_week"])
    encoder = OneHotEncoder(columns=["workclass", "education"])
    
    dataset = preprocessor.fit_transform(dataset)
    dataset = encoder.fit_transform(dataset)
    
    train_ds, valid_ds = dataset.train_test_split(test_size=0.2)
    
    def train_func():
        train_shard = ray.train.get_dataset_shard("train")
        valid_shard = ray.train.get_dataset_shard("valid")
        
        train_df = train_shard.to_pandas()
        valid_df = valid_shard.to_pandas()
        
        train_X, train_y = train_df.drop("income", axis=1), train_df["income"]
        valid_X, valid_y = valid_df.drop("income", axis=1), valid_df["income"]
        
        train_set = lgb.Dataset(train_X, label=train_y)
        valid_set = lgb.Dataset(valid_X, label=valid_y)
        
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "verbose": -1,
        }
        
        model = lgb.train(
            params,
            train_set,
            valid_sets=[valid_set],
            num_boost_round=100,
            callbacks=[RayTrainReportCallback()]
        )
    
    trainer = LightGBMTrainer(
        train_func,
        scaling_config=ScalingConfig(num_workers=2),
        datasets={"train": train_ds, "valid": valid_ds},
        metadata={"preprocessor": preprocessor.serialize(), "encoder": encoder.serialize()}
    )
    
    return trainer.fit()
```

## Batch Inference

After training, you can use the model for distributed batch inference:

```python
class LightGBMPredictor:
    def __init__(self, checkpoint):
        from ray.train.lightgbm import RayTrainReportCallback
        self.model = RayTrainReportCallback.get_model(checkpoint)
        
        # Load preprocessors if available
        metadata = checkpoint.get_metadata()
        if "preprocessor" in metadata:
            from ray.data.preprocessors import StandardScaler
            self.preprocessor = StandardScaler.deserialize(metadata["preprocessor"])
        else:
            self.preprocessor = None
    
    def __call__(self, batch):
        if self.preprocessor:
            batch = self.preprocessor.transform_batch(batch)
        
        predictions = self.model.predict(batch)
        return {"predictions": predictions}

# Apply predictions to test data
test_dataset = ray.data.read_csv("path/to/test.csv")
predictions = test_dataset.map_batches(
    LightGBMPredictor,
    fn_constructor_args=[result.checkpoint],
    batch_format="pandas"
)
```

## Hyperparameter Tuning

Integrate with Ray Tune for automated hyperparameter optimization:

```python
from ray import tune
from ray.tune import Tuner

def create_tune_trainer():
    dataset = ray.data.read_csv("s3://anonymous@air-example-data/breast_cancer.csv")
    train_ds, valid_ds = dataset.train_test_split(test_size=0.2)
    
    def train_func(config):
        train_shard = ray.train.get_dataset_shard("train")
        valid_shard = ray.train.get_dataset_shard("valid")
        
        train_df = train_shard.to_pandas()
        valid_df = valid_shard.to_pandas()
        
        train_X, train_y = train_df.drop("target", axis=1), train_df["target"]
        valid_X, valid_y = valid_df.drop("target", axis=1), valid_df["target"]
        
        train_set = lgb.Dataset(train_X, label=train_y)
        valid_set = lgb.Dataset(valid_X, label=valid_y)
        
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "num_leaves": config["num_leaves"],
            "learning_rate": config["learning_rate"],
            "feature_fraction": config["feature_fraction"],
            "verbose": -1,
        }
        
        model = lgb.train(
            params,
            train_set,
            valid_sets=[valid_set],
            num_boost_round=100,
            callbacks=[RayTrainReportCallback()]
        )
    
    trainer = LightGBMTrainer(
        train_func,
        scaling_config=ScalingConfig(num_workers=2),
        datasets={"train": train_ds, "valid": valid_ds}
    )
    
    return trainer

# Create tuner
tuner = Tuner(
    create_tune_trainer(),
    param_space={
        "train_loop_config": {
            "num_leaves": tune.randint(10, 100),
            "learning_rate": tune.loguniform(0.01, 0.3),
            "feature_fraction": tune.uniform(0.5, 1.0),
        }
    },
    tune_config=tune.TuneConfig(
        metric="eval-binary_logloss",
        mode="min",
        num_samples=10,
    )
)

# Run hyperparameter tuning
results = tuner.fit()
best_result = results.get_best_result()
```

## Best Practices

1. **Resource Configuration**: Allocate appropriate CPU/GPU resources based on your data size and cluster capacity
2. **Data Preprocessing**: Use Ray Data preprocessors for efficient data transformation
3. **Checkpointing**: Enable checkpointing for fault tolerance and model persistence
4. **Early Stopping**: Use LightGBM's built-in early stopping to prevent overfitting
5. **Monitoring**: Leverage Ray Train's built-in metrics reporting for training monitoring
6. **Scaling**: Start with fewer workers and scale up based on performance needs

## Common Issues and Solutions

### Memory Issues
- Reduce `num_leaves` or `max_depth` parameters
- Use `feature_fraction` and `bagging_fraction` to reduce memory usage
- Consider using more workers with smaller datasets per worker

### Slow Training
- Ensure your data is properly sharded across workers
- Use GPU acceleration when available
- Optimize LightGBM parameters for your specific use case

### Convergence Issues
- Adjust learning rate and number of boosting rounds
- Use proper evaluation metrics for your problem type
- Consider data quality and feature engineering

## Additional Resources

- [Ray Train Documentation](https://docs.ray.io/en/latest/train/train.html)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [Ray Data Documentation](https://docs.ray.io/en/latest/data/data.html)
- [Ray Tune Documentation](https://docs.ray.io/en/latest/tune/index.html)