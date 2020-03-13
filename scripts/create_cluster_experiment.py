from pathlib import Path

from active_meta_learning import project_parameters
from hpc_cluster.utils import ArrayJob

param_dict = {"d": [10], "k": [100], "s2": [0.01]}

array_job = ArrayJob(
    param_dict=param_dict,
    working_dir=project_parameters.PROJECT_DIR,
    source_path=Path(project_parameters.PROJECT_DIR / "project.source"),
    script_path=project_parameters.SCRIPTS_DIR
    / "loss_learning_curves_maml_hypercuve_with_k_vertex_gaussians.py",
    job_submission_files_dir=project_parameters.JOB_SUBMISSION_DIR,
    job_output_dir=project_parameters.JOB_OUTPUT_DIR,
    program="python3",
    tmem=3,
    h_vmem=3,
    h_rt=3600 * 4,
    gpu=False,
)
