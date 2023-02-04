#!/bin/bash

mkdir -p deps
cd deps

if [ ! -d "mujoco210" ]; then
  wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz
  tar -xvf mujoco210-linux-x86_64.tar.gz
  rm mujoco210-linux-x86_64.tar.gz
  wget -O mujoco210/mjkey.txt https://www.roboti.us/file/mjkey.txt
else
  echo "mujoco is already installed locally"
fi

export MUJOCO_PY_MUJOCO_PATH="$(pwd)/mujoco210"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/mujoco210/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia

export MUJOCO_PY_MJKEY_PATH=$(pwd)/mujoco210/mjkey.txt
export MUJOCO_PY_MJPRO_PATH=$(pwd)/mujoco210/

if ! dpkg -s libglew-dev > /dev/null 2>&1 ; then
  if [ ! -d "$(pwd)/libglew-dev" ]; then
    apt-get download libglew-dev
    dpkg -x libglew-dev*.deb libglew-dev
    rm libglew-dev*.deb
  else
    echo "libglew-dev is already installed locally"
  fi
else
  echo "libglew-dev is already installed"
fi

if [ -d "$(pwd)/libglew-dev" ]; then
  export C_INCLUDE_PATH=$C_INCLUDE_PATH:$(pwd)/libglew-dev/usr
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(pwd)/libglew-dev/usr/lib
fi

if ! dpkg -s patchelf > /dev/null 2>&1 ; then
  if [ ! -d "$(pwd)/patchelf" ]; then
    apt-get download patchelf
    dpkg -x patchelf*.deb patchelf
    rm patchelf*.deb
  else
    echo "patchelf is already installed locally"
  fi
else
  echo "patchelf is already installed"
fi

if [ -d "$(pwd)/patchelf" ]; then
  export PATH=$PATH:$(pwd)/patchelf/usr/bin
fi

cd ..
