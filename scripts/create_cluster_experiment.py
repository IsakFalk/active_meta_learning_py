from pathlib import Path

from active_meta_learning import project_parameters
from hpc_cluster.utils import ArrayJob

param_dict = {
    "d": [3, 5, 10],
    "k": [2, 5, 8],
    "noise_w": [0.01, 0.1, 1.0],
    "num_prototypes": [5, 10]
}

array_job = ArrayJob(
    param_dict=param_dict,
    working_dir=project_parameters.PROJECT_DIR,
    source_path=Path(project_parameters.PROJECT_DIR / "project.source"),
    script_path=project_parameters.SCRIPTS_DIR
    / "1_nn_conditional_active_meta_learning_only_bias.py",
    job_submission_files_dir=project_parameters.JOB_SUBMISSION_DIR,
    job_output_dir=project_parameters.JOB_OUTPUT_DIR,
    program="python3",
    tmem=8,
    h_vmem=8,
    h_rt=3600 * 4,
    gpu=False,
)
