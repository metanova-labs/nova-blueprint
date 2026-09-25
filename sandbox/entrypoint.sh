#!/bin/sh
set -eu
db_dir=/usr/local/lib/python3.12/site-packages/nova_miner/combinatorial_db
if [ -z "$(ls -A "$db_dir")" ]; then
    cp -R /opt/combinatorial_db/. "$db_dir"/
fi
exec python /workspace/miner.py "$@"
