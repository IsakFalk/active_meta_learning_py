from pathlib import Path

PROJECT_DIR = Path(__file__).parent
SCRIPTS_DIR = PROJECT_DIR / "scripts"
JOB_SUBMISSION_DIR = PROJECT_DIR / "cluster_job_submissions"
DATA_DIR = PROJECT_DIR / "data"
INTERIM_DATA_DIR = DATA_DIR / "interim"
JOB_OUTPUT_DIR = DATA_DIR / "job_output"
