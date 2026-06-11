#!/bin/bash


# module avail node

# 1. 
module load gcc/8.4.0-gcc-10.3.0
module load cmake/3.22.2-gcc-10.3.0


# 2. ROOT
## install ROOT:
## - script:  /public/home/liuz1/work/26.03.18_rest/NvDEX_package/scripts/installation/installROOT.sh
##   - modification:  ROOT_VERSION=6.20.00
##   - modification:  disable MTVA: -Dtmva=OFF 
## - run: ./installROOT.sh 

source /public/home/liuz1/apps/root-6.20.00/install/bin/thisroot.sh


# 3. Geant4

## /public/home/liuz1/apps/

# 3.1 Xerces-C
## wget https://archive.apache.org/dist/xerces/c/3/sources/xerces-c-3.2.3.tar.gz
## cmake -DCMAKE_INSTALL_PREFIX=/public/home/liuz1/apps/xerces-c-install ..
## make -j8 install

# 3.2 Geant4
## cmake -DCMAKE_INSTALL_PREFIX=../install \
##       -DGEANT4_INSTALL_DATA=ON \
##       -DGEANT4_BUILD_MULTITHREADED=ON \
##       -DGEANT4_USE_GDML=ON \
##       -DXERCESC_ROOT_DIR=/public/home/liuz1/apps/xerces-c-install \
##        ..
##
## make -j8 
## make install
source /public/home/liuz1/apps/geant4-v10.2.3/install/bin/geant4.sh


# 4. garfield6
export Garfield_DIR=/public/home/liuz1/apps/garfield6/install
export LD_LIBRARY_PATH=$Garfield_DIR/lib:$LD_LIBRARY_PATH
export HEED_DATABASE=/public/home/liuz1/apps/garfield6/Heed/heed++/database
#export CMAKE_PREFIX_PATH=/public/home/liuz1/apps/root-6.20.00:/public/home/liuz1/apps/geant4-v10.2.3/install:/public/home/liuz1/apps/garfield6-install:$CMAKE_PREFIX_PATH


# 4. REST
## cd build
## rm -f CMakeCache.txt
### # cmake .. -DCMAKE_INSTALL_PREFIX=../install/master/ 
### # cmake .. -DCMAKE_INSTALL_PREFIX=../install/master/  -DREST_G4=ON -DREST_DECAY0=ON -DREST_DATABASE=ON
### cmake .. -DCMAKE_INSTALL_PREFIX=../install/master/  -DREST_G4=ON 
### cmake .. -DCMAKE_INSTALL_PREFIX=../install/master/  -DREST_G4=ON -DREST_DECAY0=OFF -DREST_DATABASE=OFF
# 
# cmake .. -Wno-dev \
#     -DCMAKE_INSTALL_PREFIX=../install/master/ \
#     -DREST_WELCOME=OFF \
#     -DREST_G4=ON \
#     -DREST_GARFIELD=ON \
#     -DREST_DATABASE=OFF \
#     -DREST_DECAY0=OFF \
#     -DCMAKE_PREFIX_PATH="/public/home/liuz1/apps/garfield6/install" \
#     -DGarfield_INCLUDE_DIRS="/public/home/liuz1/apps/garfield6/install/include/Garfield"
# 
# make -j4 install

source /public/home/liuz1/work/26.03.18_rest/NvDEX_package/install/thisREST.sh


# 5. Print
echo "========================================"
echo "  - REST Environment Loaded:  "
echo "    - GCC: $(gcc -dumpversion)"
echo "    - CMake: $(cmake --version | head -n 1)"
echo "    - ROOT  : $(root-config --version)  ($(which root))"
echo "    - Geant4 : $(geant4-config --version)  ($(which geant4-config))"
echo "    - garfield6 : $Garfield_DIR"
REST_VER=$(grep "REST_RELEASE " $REST_PATH/include/TRestVersion.h | awk '{print $3}' | tr -d '"')
echo "    - REST Version: $REST_VER"
echo "========================================"





