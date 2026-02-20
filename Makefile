PACKAGE_NAME = img2art_search

lint:
	isort ${PACKAGE_NAME}
	black ${PACKAGE_NAME}
	flake8 ${PACKAGE_NAME}
	mypy ${PACKAGE_NAME}
viz:
	poetry run python3 main.py interface
train:
	poetry run python3 main.py train
wikiart:
	poetry run python3 main.py gallery
build-image:
	docker build -t img2art-search .
run-on-docker:
	docker run --env-file .env -p 7860:7860 img2art-search
serve:
	poetry run uvicorn fastapi_app:app --host 0.0.0.0 --port 7860 --reload