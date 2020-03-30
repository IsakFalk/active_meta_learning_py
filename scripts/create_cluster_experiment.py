from pathlib import Path

from active_meta_learning import project_parameters
from hpc_cluster.utils import ArrayJob

param_dict = {
    "d": [5, 10, 30, 50],
    "k": [4, 10],
    "s2_w": [0.001],
    "noise_y_high": [0.5, 1.0]
}

array_job = ArrayJob(
    param_dict=param_dict,
    working_dir=project_parameters.PROJECT_DIR,
    source_path=Path(project_parameters.PROJECT_DIR / "project.source"),
    script_path=project_parameters.SCRIPTS_DIR
    / "hi_low_output_noise_mixture_idea_with_low_k_shot.py",
    job_submission_files_dir=project_parameters.JOB_SUBMISSION_DIR,
    job_output_dir=project_parameters.JOB_OUTPUT_DIR,
    program="python3",
    tmem=8,
    h_vmem=8,
    h_rt=3600 * 4,
    gpu=False,
)
