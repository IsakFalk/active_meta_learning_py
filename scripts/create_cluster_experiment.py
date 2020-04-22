from pathlib import Path

from active_meta_learning import project_parameters
from hpc_cluster.utils import ArrayJob

param_dict = {
    "seed": list(range(10)),
    "k_nn": [5],
    "env_name": ["hypersphere-d=10.hkl"],
}

array_job = ArrayJob(
    param_dict=param_dict,
    working_dir=project_parameters.PROJECT_DIR,
    source_path=Path(project_parameters.PROJECT_DIR / "project.source"),
    script_path=project_parameters.SCRIPTS_DIR / "meta_knn" / "ridge",
    job_submission_files_dir=project_parameters.JOB_SUBMISSION_DIR,
    job_output_dir=project_parameters.JOB_OUTPUT_DIR,
    program="python3",
    tmem=8,
    h_vmem=8,
    h_rt=3600 * 4,
    gpu=False,
)
