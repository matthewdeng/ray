# flake8: noqa
# isort: skip_file

# __lightgbm_start__
import pandas as pd
import lightgbm as lgb

# 1. Load your data as a `lightgbm.Dataset`.
train_df = pd.read_csv("s3://ray-example-data/iris/train/1.csv")
eval_df = pd.read_csv("s3://ray-example-data/iris/val/1.csv")

train_X = train_df.drop("target", axis=1)
train_y = train_df["target"]
eval_X = eval_df.drop("target", axis=1)
eval_y = eval_df["target"]

dtrain = lgb.Dataset(train_X, label=train_y)
deval = lgb.Dataset(eval_X, label=eval_y)

# 2. Define your LightGBM model training parameters.
params = {
    "objective": "regression",
    "metric": "l2",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
}

# 3. Do non-distributed training.
bst = lgb.train(
    params,
    train_set=dtrain,
    valid_sets=[deval],
    valid_names=["validation"],
    num_boost_round=100,
)
# __lightgbm_end__


# __lightgbm_ray_start__
import lightgbm as lgb

import ray.train
from ray.train.lightgbm import LightGBMTrainer, RayTrainReportCallback

# 1. Load your data as a Ray Data Dataset.
train_dataset = ray.data.read_csv("s3://anonymous@ray-example-data/iris/train")
eval_dataset = ray.data.read_csv("s3://anonymous@ray-example-data/iris/val")


def train_func():
    # 2. Load your data shard as a `lightgbm.Dataset`.

    # Get dataset shards for this worker
    train_shard = ray.train.get_dataset_shard("train")
    eval_shard = ray.train.get_dataset_shard("eval")

    # Convert shards to pandas DataFrames
    train_df = train_shard.materialize().to_pandas()
    eval_df = eval_shard.materialize().to_pandas()

    train_X = train_df.drop("target", axis=1)
    train_y = train_df["target"]
    eval_X = eval_df.drop("target", axis=1)
    eval_y = eval_df["target"]

    dtrain = lgb.Dataset(train_X, label=train_y)
    deval = lgb.Dataset(eval_X, label=eval_y)

    # 3. Define your LightGBM model training parameters.
    params = {
        "objective": "regression",
        "metric": "l2",
        "boosting_type": "gbdt",
        "num_leaves": 31,
        "learning_rate": 0.05,
        "feature_fraction": 0.9,
        # Add network parameters for distributed training
        **ray.train.lightgbm.get_network_params(),
    }

    # 4. Do distributed data-parallel training.
    # Ray Train sets up the necessary coordinator processes and
    # environment variables for your workers to communicate with each other.
    bst = lgb.train(
        params,
        train_set=dtrain,
        valid_sets=[deval],
        valid_names=["validation"],
        num_boost_round=100,
        # Optional: Use the `RayTrainReportCallback` to save and report checkpoints.
        callbacks=[RayTrainReportCallback()],
    )


# 5. Configure scaling and resource requirements.
scaling_config = ray.train.ScalingConfig(num_workers=2, resources_per_worker={"CPU": 2})

# 6. Launch distributed training job.
trainer = LightGBMTrainer(
    train_func,
    scaling_config=scaling_config,
    datasets={"train": train_dataset, "eval": eval_dataset},
    # If running in a multi-node cluster, this is where you
    # should configure the run's persistent storage that is accessible
    # across all worker nodes.
    # run_config=ray.train.RunConfig(storage_path="s3://..."),
)
result = trainer.fit()

# 7. Load the trained model
import os

with result.checkpoint.as_directory() as checkpoint_dir:
    model_path = os.path.join(checkpoint_dir, RayTrainReportCallback.CHECKPOINT_NAME)
    model = lgb.Booster(model_file=model_path)
# __lightgbm_ray_end__