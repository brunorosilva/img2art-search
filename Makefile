lint:
	isort img2art_search
	black img2art_search
	flake8 img2art_search
viz:
	poetry run python3 main.py interface
train:
	poetry run python3 main.py train
wikiart:
	poetry run python3 main.py gallery