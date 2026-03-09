# mmds

Minimal `uv`-managed Python app scaffold intended to run inside Docker.

## Local project files

- `pyproject.toml`: Python project metadata
- `src/mmds`: application package and `mmds` console entrypoint
- `docker-compose.yml`: development container definition
- `Dockerfile`: Ubuntu-based GPU-capable development image

## Development workflow

Build the image:

```bash
docker compose build
```

Start the development container:

```bash
docker compose up -d
```

Open a shell in the running container:

```bash
docker compose exec mmds bash
```

Run the sample app inside the container:

```bash
docker compose exec mmds uv run mmds
```

## Notes

- The development container exposes `codex`, `claude`, and `cursor-agent`.
- The repo keeps a project-local `.venv`. If you switch between host and container execution, rerun `uv sync` in the environment you want to use because the virtualenv is not portable across both contexts.
