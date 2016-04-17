#!/bin/sh
# get file path
cwd=`dirname "${0}"`
expr "${0}" : "/.*" > /dev/null || cwd=`(cd "${cwd}" && pwd)`

g++ ${cwd}/sample1.cpp -I /usr/include/ -lhdf5 -o test && ${cwd}/test

