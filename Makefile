reinstall_package:
	@pip uninstall -y cooking_recipes || :
	@pip install -e .
run_preprocess:
	python -c 'from api_folter/cooking_recipes.interface.main import preprocess; preprocess()'

run_train:
	python -c 'from api_folter/cooking_recipes.interface.main import train; train()'

run_pred:
	python -c 'from api_folter/cooking_recipes.interface.main import pred; pred()'

run_evaluate:
	python -c 'from api_folter/cooking_recipes.interface.main import evaluate; evaluate()'

run_all: run_preprocess run_train run_pred run_evaluate
