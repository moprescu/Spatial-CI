# This file is used to generate all the baselines for the paper.

from omegaconf import OmegaConf

conda: "benchmarks/conda.yaml"


# == Load configs ==
if len(config) == 0:
    raise Exception(
        "No config file passed to snakemake."
        " Use flag --configfile benchmarks/conf/pipeline.yaml"
    )


# make target files
targets = []
for t_type in ("disc", "cont"):
    envs = config["spaceenvs"][t_type]
    algos = config["algorithms"].get(t_type) or []
    for env in envs:
        for algo in algos:
            logfile = f"{config['logdir']}/{env}/{algo}.jsonl"
            targets.append(logfile)


# == Define rules ==
rule all:
    input:
        targets,


rule train_spaceenv:
    output:
        config["logdir"] + "/{spaceenv}/{algo}.jsonl",
    threads: config["concurrency"] * config["cpus_per_task"]
    resources:
        mem_mb=config["mem_mb"],
    params:
        concurrency=config["concurrency"],
        overwrite=config["overwrite"],
    log:
        err="logs/{spaceenv}/{algo}.err",
    shell:
        """
        export LD_LIBRARY_PATH=/home/idies/workspace/Temporary/akhot2/scratch/spacedata/lib/python3.10/site-packages/nvidia/cuda/lib:$LD_LIBRARY_PATH
        HYDRA_FULL_ERROR=1 python benchmarks/run.py \
            algo={wildcards.algo} \
            spaceenv={wildcards.spaceenv} \
            concurrency={params.concurrency} \
            overwrite={params.overwrite} \
            hydra.run.dir=logs/{wildcards.spaceenv}/{wildcards.algo} \
            2> {log.err}
        """
