#!/bin/sh

for x in 100 200 500 1000 2000; do
    for k in 1 2 5 10; do
        ./mf-benchmark $x $x $k || exit 1
    done
done
