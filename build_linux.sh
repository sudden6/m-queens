#! /bin/bash

# don't build in-tree
mkdir -p ./build/
cd ./build/

BUILD_DIR="$PWD"

# get BOINC source
git clone --depth 1 -b client_release/7.14/7.14.2 https://github.com/BOINC/boinc "$BUILD_DIR/boinc"

# output dir for boinc static build
mkdir -p ./static_boinc/

# build boinc
cd ./boinc
./_autosetup
./configure --disable-server --disable-manager --disable-fcgi --disable-client --prefix="$BUILD_DIR/static_boinc/"

make -j4
make install

# workaround to support BOINC out of tree build
cp ./config.h "$BUILD_DIR/static_boinc/include/boinc"
cp ./project_specific_defines.h "$BUILD_DIR/static_boinc/include/boinc"

# build m-queens
cd ..
mkdir ./m-queens
cd ./m-queens
cmake -DCMAKE_BUILD_TYPE=Release -DNO_BOINC=OFF -DBUILD_STATIC=ON -DBOINC_ONLY=ON -DBOINC_STATIC_DIR=../static_boinc  "$BUILD_DIR/../"
make -j4
