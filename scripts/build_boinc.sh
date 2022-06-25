mkdir -p /c/build/
#rm -R /c/build/boinc/
#cd /c/build/
#git clone https://github.com/BOINC/boinc.git --branch client_release/7.18/7.18.1 boinc
cd /c/build/boinc
./_autosetup
./configure --disable-server --disable-manager --disable-fcgi --disable-client --prefix=/c/build/static_boinc/
make -j 4
make install