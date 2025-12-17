export TAG = latest
export COMPOSE_DOCKER_CLI_BUILD=1
export DOCKER_BUILDKIT=1
export COMPOSE_NAME=reconhecimento_facial

ruff:
	ruff format . && ruff check . --fix

docker-build-up-compose:
	docker-compose -f docker-compose.yml -p $(COMPOSE_NAME) up --build -d
