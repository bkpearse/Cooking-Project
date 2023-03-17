.DEFAULT_GOAL := default
#################### PACKAGE ACTIONS ###################
reinstall_package:
	@pip uninstall -y project_cook || :
	@pip install -e .

run_api:
	uvicorn project_cook.api_folder.api_file:api --reload

reset_all_files: reset_local_files reset_bq_files reset_gcs_files

# run_preprocess:
# 	python -c 'from api_folder/project_cook.interface.main import preprocess; preprocess()'

# run_train:
# 	python -c 'from api_folder/project_cook.interface.main import train; train()'

# run_pred:
# 	python -c 'from api_folder/project_cook.interface.main import pred; pred()'

# run_evaluate:
# 	python -c 'from api_folder/project_cook.interface.main import evaluate; evaluate()'

# run_all: run_preprocess run_train run_pred run_evaluate
