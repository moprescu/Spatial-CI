import logging
import os
import shutil
import time

import hydra
import jsonlines
import ray
from ray import train
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything
from ray import tune
import torch, gc
import numpy as np
import pandas as pd

import sci
from sci import SpaceEnv
from sci.algorithms.datautils import spatial_train_test_split

LOGGER = logging.getLogger(__name__)
MAX_TRIALS = 3

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    logfile = cfg.tunefile
    os.makedirs(os.path.dirname(logfile), exist_ok=True)
    LOGGER.info(f"Logging to {logfile}")
    seed_everything(0)

    # check if logfile exists and overwrite, delete it and continue
    # otherwise return
    if os.path.exists(logfile):
        if cfg.overwrite:
            LOGGER.info(f"Cleaning logfile {logfile}")
            os.remove(logfile)
        else:
            LOGGER.info(f"Logfile {logfile} already exists, skipping")
            return

    # check if run_config.storage_path exists, and if so, delete it
    raydir = f"{cfg.algo.tune.run_config.storage_path}/{cfg.algo.name}"
    cfg.algo.tune.run_config.storage_path = os.path.abspath(cfg.algo.tune.run_config.storage_path)
    if os.path.exists(raydir):
        LOGGER.info(f"Cleaning ray path {raydir}")
        shutil.rmtree(raydir)

    env_name = cfg.spaceenv 
    env = SpaceEnv(env_name, dir="downloads", algo_rad=0 if not cfg.algo.use_interference else cfg.algo.method.radius)
    # print(env.coordinates)

    train_ix, test_ix, _ = spatial_train_test_split(
        env.graph, **{**cfg.spatial_train_test_split, "buffer": cfg.spatial_train_test_split["buffer"] + env.radius}
    )
    right_method = "deconfounder" in cfg.algo.name or "spatial" in cfg.algo.name or "gcn" in cfg.algo.name
    # train_ix, test_ix, _ = spatial_train_test_split(
    #     env.graph, **{**cfg.spatial_train_test_split, "buffer": cfg.spatial_train_test_split["buffer"] + env.radius if cfg.algo.use_interference else cfg.spatial_train_test_split["buffer"]}
    # )
    for i, full_dataset in enumerate(env.make_all()):
        LOGGER.info(f"Running dataset {i} from {env_name}")

        # train/test split
        LOGGER.info("...splitting dataset into train/test")
        if cfg.algo.needs_train_test_split:
            train_dataset = full_dataset[train_ix]
            test_dataset = full_dataset[test_ix]
        else:
            train_dataset = full_dataset
            test_dataset = full_dataset

        # setup hyperparameter tuning objective
        param_space = dict(hydra.utils.instantiate(cfg.algo.tune.param_space))
        if len(param_space) > 0:

            def objective(config):
                # Creates same seed for same hyperparameter combination
                config_str = str(sorted(config.items()))
                seed = hash(f"{config_str}_{i}") % (2**32)
                seed_everything(seed)
    
                method = hydra.utils.instantiate(cfg.algo.method, **config)
                method.fit(train_dataset)
                tune_metric = method.tune_metric(test_dataset)
                num_trials = 0

                while right_method and (tune_metric > 100 or np.isnan(tune_metric)) and num_trials < MAX_TRIALS:
                    if cfg.algo.needs_train_test_split:
                        tmp_train_ix, tmp_test_ix, _ = spatial_train_test_split(
                            env.graph, **{**cfg.spatial_train_test_split, "buffer": cfg.spatial_train_test_split["buffer"] + env.radius}
                        )
                        tmp_train_dataset = full_dataset[tmp_train_ix]
                        tmp_test_dataset = full_dataset[tmp_test_ix]
                    else:
                        tmp_train_dataset = train_dataset
                        tmp_test_dataset = test_dataset
                    method = hydra.utils.instantiate(cfg.algo.method, **config)
                    method.fit(tmp_train_dataset)
                    tune_metric = method.tune_metric(tmp_test_dataset)
                    num_trials = num_trials + 1

                    # del method
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.ipc_collect()
                    
                effects = method.eval(full_dataset)
                evaluator = sci.DatasetEvaluator(full_dataset)
                eval_results = evaluator.eval(**effects)
                eval_results = {k: eval_results.get(k, None) for k in ("ate", "erf", "ite", "spill")}
                eval_results["_metric"] = tune_metric
                tune.report(eval_results)

            objective = tune.with_resources(objective, dict(cfg.algo.tune.resources))

            # create tuner
            LOGGER.info("...setting up hyperparameter tuning")
            tune_config = hydra.utils.instantiate(cfg.algo.tune.tune_config)
            run_config = hydra.utils.instantiate(
                cfg.algo.tune.run_config, name=f"{i:02d}"
            )
            ray.shutdown()
            ray.init(
                ignore_reinit_error=True,
                include_dashboard=False,
                # num_cpus=cfg.concurrency,
            )
            tuner = tune.Tuner(
                objective,
                tune_config=tune_config,
                param_space=param_space,
                run_config=run_config,
            )

            # run hyperparameter tuning
            LOGGER.info("...running hyperparameter tuning")
            results = tuner.fit()

            # save all grid results in all_results
            best_params = results.get_best_result().config
            LOGGER.info(f"Best params: {best_params}")
            
            df = results.get_dataframe()
            print(df.columns)
            df['dataset_index'] = i
            df['algo'] = cfg.algo.name
            df['env'] = env_name
            
            if os.path.exists(logfile):
                existing_df = pd.read_parquet(logfile)
                df = pd.concat([existing_df, df], ignore_index=True)
            df.to_parquet(logfile)

if __name__ == "__main__":
    main()
