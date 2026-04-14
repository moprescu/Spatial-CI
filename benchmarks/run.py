import logging
import os
import pickle
import shutil
import time

import hydra
import jsonlines
import ray
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import seed_everything
from ray import tune
import torch, gc

import sci
from sci import SpaceEnv, ArcticEnv
from sci.algorithms.datautils import spatial_train_test_split

LOGGER = logging.getLogger(__name__)
MAX_TRIALS = 3

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    # make logfilealgo
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    logfile = cfg.logfile
    os.makedirs(os.path.dirname(logfile), exist_ok=True)
    LOGGER.info(f"Logging to {logfile}")

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
    if env_name.startswith("arctic"):
        env = ArcticEnv(env_name, dir="downloads", algo_rad=0 if not cfg.algo.use_interference else cfg.algo.method.radius)
    else:
        env = SpaceEnv(env_name, dir="downloads", algo_rad=0 if not cfg.algo.use_interference else cfg.algo.method.radius)
    # print(env.coordinates)
        
    for gseed in cfg.global_seeds:
        torch.cuda.empty_cache()
        seed_everything(gseed)

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

                # Extract small objects from env so the closure doesn't
                # capture the entire (potentially huge) env object.
                _env_graph = env.graph
                _env_radius = env.radius
                _split_kwargs = {**cfg.spatial_train_test_split, "buffer": cfg.spatial_train_test_split["buffer"] + env.radius}

                def objective(config, _train_ref=None, _test_ref=None, _full_ref=None):
                    _train_dataset = ray.get(_train_ref)
                    _test_dataset = ray.get(_test_ref)
                    _full_dataset = ray.get(_full_ref)

                    seed_everything(gseed)
                    method = hydra.utils.instantiate(cfg.algo.method, **config)
                    method.fit(_train_dataset)
                    tune_metric = method.tune_metric(_test_dataset)
                    num_trials = 0

                    while right_method and tune_metric > 100 and num_trials < MAX_TRIALS:
                        if cfg.algo.needs_train_test_split:
                            tmp_train_ix, tmp_test_ix, _ = spatial_train_test_split(
                                _env_graph, **_split_kwargs
                            )
                            tmp_train_dataset = _full_dataset[tmp_train_ix]
                            tmp_test_dataset = _full_dataset[tmp_test_ix]
                        else:
                            tmp_train_dataset = _train_dataset
                            tmp_test_dataset = _test_dataset
                        method = hydra.utils.instantiate(cfg.algo.method, **config)
                        method.fit(tmp_train_dataset)
                        tune_metric = method.tune_metric(tmp_test_dataset)
                        num_trials = num_trials + 1

                        del method
                        gc.collect()
                        torch.cuda.empty_cache()
                        torch.cuda.reset_peak_memory_stats()
                        torch.cuda.ipc_collect()

                    return tune_metric

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
                # Put large datasets in Ray object store to avoid
                # serializing them into every worker closure.
                train_ref = ray.put(train_dataset)
                test_ref = ray.put(test_dataset)
                full_ref = ray.put(full_dataset)

                from functools import partial
                objective_with_data = partial(objective, _train_ref=train_ref, _test_ref=test_ref, _full_ref=full_ref)
                objective_with_data = tune.with_resources(objective_with_data, dict(cfg.algo.tune.resources))

                tuner = tune.Tuner(
                    objective_with_data,
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
            else:
                LOGGER.info("...skipping hyperparameter tuning, no param space provided")
                best_params = {}
                
            if len(param_space) > 0:
                ray.shutdown()
                time.sleep(2)

            LOGGER.info("...training full model")
            seed_everything(gseed)
            torch.cuda.empty_cache()
            method = hydra.utils.instantiate(cfg.algo.method, **best_params)
            method.fit(full_dataset)
            if len(param_space) > 0:
                tune_metric = method.tune_metric(test_dataset)
                num_trials = 0
                while right_method and tune_metric > 100 and num_trials < MAX_TRIALS:
                    method = hydra.utils.instantiate(cfg.algo.method, **best_params)
                    method.fit(full_dataset)
                    tune_metric = method.tune_metric(test_dataset)
                    num_trials = num_trials + 1
            effects = method.eval(full_dataset)

            # save the final trained model
            model_path = logfile.replace(".jsonl", f"_i{i}_seed{gseed}_model.pkl")
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            with open(model_path, "wb") as f:
                pickle.dump(method, f)
            LOGGER.info(f"Saved model to {model_path}")

            # method.plot_latent(full_dataset, logfile.replace(".jsonl", f"_i{i}_seed{gseed}.pdf"))
            # LOGGER.info(f"actual effects: {effects}")
            
            # del method
            # gc.collect()
            # torch.cuda.empty_cache()
            # torch.cuda.reset_peak_memory_stats()
            # torch.cuda.ipc_collect()
                    
            # load evaluator
            evaluator = sci.DatasetEvaluator(full_dataset)
            eval_keys = {"ate", "att", "atc", "ite", "erf", "spill"}
            eval_results = evaluator.eval(**{k: v for k, v in effects.items() if k in eval_keys})
            eval_results = {k: float(eval_results[k]) if eval_results.get(k) is not None else None for k in ("ate", "erf", "ite", "spill")}
            eval_results["env"] = env_name
            eval_results["dataset_id"] = i
            eval_results["algo"] = cfg.algo.name
            eval_results["smoothness"] = full_dataset.smoothness_score
            eval_results["confounding"] = full_dataset.confounding_score
            eval_results["timestamp"] = timestamp 
            eval_results["binary"] = full_dataset.has_binary_treatment()

            eval_results["radius"] = full_dataset.radius
            eval_results["seed"] = gseed
            eval_results["miss"] = full_dataset.topfeat[i]
            if hasattr(method, "cur_val_p_value"):
                eval_results["p_value"] = method.cur_val_p_value

            # Arctic counterfactual: % SIC change from 5% LWDN reduction
            if "cf_annual_pct" in effects:
                eval_results["cf_annual_pct"] = effects["cf_annual_pct"]
            if "cf_summer_pct" in effects:
                eval_results["cf_summer_pct"] = effects["cf_summer_pct"]
            # Arctic counterfactual: % SIC change from +18 W/m² LWDN increase
            if "cf_plus18_annual_pct" in effects:
                eval_results["cf_plus18_annual_pct"] = effects["cf_plus18_annual_pct"]
            if "cf_plus18_summer_pct" in effects:
                eval_results["cf_plus18_summer_pct"] = effects["cf_plus18_summer_pct"]

            LOGGER.info(f"eval results: {eval_results}")

            with jsonlines.open(logfile, mode="a") as writer:
                writer.write(eval_results)


if __name__ == "__main__":
    main()
