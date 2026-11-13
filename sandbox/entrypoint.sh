#!/bin/sh
set -eu
python -I /usr/local/bin/sandbox-extract.py

db_dir=/usr/local/lib/python3.12/site-packages/nova_miner/combinatorial_db
if [ -z "$(ls -A "$db_dir")" ]; then
    cp -R /opt/combinatorial_db/. "$db_dir"/
fi
cd /workspace
exec python /workspace/miner.py "$@"
