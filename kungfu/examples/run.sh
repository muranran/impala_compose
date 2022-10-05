#!/bin/sh
set -e

cd $(dirname $0)

export KUNGFU_DISABLE_AUTO_LOAD=1

python3 tf1_mnist_session.py --rank 0  &
python3 tf1_mnist_session.py --rank 1  &

wait

echo done
