lint:
	isort makeitsports_bot
	black makeitsports_bot
	flake8 makeitsports_bot
viz:
	poetry run python3 main.py interface
train:
	poetry run python3 main.py train
wikiart:
	poetry run python3 main.py gallery