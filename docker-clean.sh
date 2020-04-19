#!/usr/bin/env bash
docker-compose down
docker container prune --filter='label=luigi_task_id'