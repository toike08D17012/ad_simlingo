#!/bin/bash

cd $(dirname ${BASH_SOURCE[0]})

docker compose run --rm --name simlingo simlingo
