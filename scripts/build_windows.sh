#!/bin/bash

mkdir -p _win_build
cd _win_build
rm ./* -R

cmake -G Ninja ..

cmake --build .
