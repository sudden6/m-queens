#!/bin/bash

mkdir -p _win_build
cd _win_build
rm ./* -R

cmake -G Ninja -DWITH_BOINC=ON ..

cmake --build .
