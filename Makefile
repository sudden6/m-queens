# Change the following to match your installation
BOINC_DIR = ../../static_boinc
BOINC_INC_DIR = $(BOINC_DIR)/include
BOINC_LIB_DIR = $(BOINC_DIR)/lib64

CXXFLAGS = -Ofast \
    --static -static-libgcc \
    -I ./ \
    -I ./cxxopts/include/ \
    -I$(BOINC_INC_DIR) \
    -L$(BOINC_LIB_DIR) \
    -L.

PROGS = m-queens-boinc m-queens-presolver

all: $(PROGS)

libstdc++.a:
	ln -s `g++ -print-file-name=libstdc++.a`

clean:
	rm $(PROGS) *.o

distclean:
	/bin/rm -f $(PROGS) *.o libstdc++.a
boinc/main.o:
	g++ $(CXXFLAGS) -c boinc/main.cpp
components:
	g++ $(CXXFLAGS) -c cpusolver.cpp
	g++ $(CXXFLAGS) -c presolver.cpp
	g++ $(CXXFLAGS) -c result_file.cpp
	g++ $(CXXFLAGS) -c serialize_util.cpp
	g++ $(CXXFLAGS) -c start_file.cpp

m-queens-boinc: components libstdc++.a $(BOINC_LIB_DIR)/libboinc.a $(BOINC_LIB_DIR)/libboinc_api.a
	g++ $(CXXFLAGS) -c boinc/main.cpp -o boinc_main.o
	g++ $(CXXFLAGS) -o $@ boinc_main.o cpusolver.o presolver.o result_file.o serialize_util.o \
	start_file.o libstdc++.a -Wl,-Bstatic,--whole-archive -lpthread -Wl,--no-whole-archive -lboinc_api -lboinc -static-libgcc
	strip $@

m-queens-presolver: components libstdc++.a
	g++ $(CXXFLAGS) -c presolver/main.cpp -o presolver_main.o
	g++ $(CXXFLAGS) -o $@ presolver_main.o cpusolver.o presolver.o result_file.o serialize_util.o \
	start_file.o libstdc++.a -Wl,-Bstatic,--whole-archive -lpthread -Wl,--no-whole-archive -lboinc_api -lboinc -static-libgcc
	strip $@

