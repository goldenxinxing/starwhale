#!/usr/bin/env bash

set -e

if [ "${SW_TASK_DISABLE_DEBUG}" != "1" ]; then
    set -x
fi

ulimit -n 65535 || true

CONDA_BIN="/opt/miniconda3/bin"
PIP_CACHE_DIR=${SW_PIP_CACHE_DIR:=/"${SW_USER:-root}"/.cache/pip}
PYTHON_VERSION=${SW_RUNTIME_PYTHON_VERSION:-"3.8"}
RUNTIME_RESTORED=${SW_USER_RUNTIME_RESTORED:-0}

welcome() {
    echo "************************************"
    echo "StarWhale Base Entrypoint"
    echo "Date: `date -u +%Y-%m-%dT%H:%M:%SZ`"
    echo "Starwhale Version: ${SW_VERSION}"
    echo "Python Version: ${PYTHON_VERSION}"
    echo "Runtime Restored: ${RUNTIME_RESTORED}"
    echo "Command type(Whether use custom command): ${USE_CUSTOM_CMD}"
    echo "Run: $1"
    echo "************************************"
}

set_python_alter() {
    echo "-->[Preparing] set python/python3 to $PYTHON_VERSION ..."
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python"$PYTHON_VERSION" 10
    update-alternatives --install /usr/bin/python python /usr/bin/python"$PYTHON_VERSION" 10
    python3 --version
}

set_pip_config() {
    echo "-->[Preparing] config pypi and conda config ..."

    if [ "${SW_PYPI_INDEX_URL}" ] ; then
        echo -e "\t ** use SW_PYPI_* env to config ~/.pip/pip.conf"
        mkdir -p ~/.pip
        cat > ~/.pip/pip.conf << EOF
[global]
index-url = ${SW_PYPI_INDEX_URL}
extra-index-url = ${SW_PYPI_EXTRA_INDEX_URL}
timeout = ${SW_PYPI_TIMEOUT:-90}

[install]
trusted-host= ${SW_PYPI_TRUSTED_HOST}
EOF
        echo -e "\t ** current pip conf:"
        echo "-------------------"
        cat ~/.pip/pip.conf
        echo "-------------------"
    else
        echo -e "\t ** use image builtin pip.conf"
    fi

    if [ -n "$SW_CONDA_CONFIG" ] ; then
      echo -e "\t ** use SW_CONDA_CONFIG env to config ~/.condarc"
      echo "$SW_CONDA_CONFIG" > ~/.condarc
      echo -e "\t ** current .condarc:"
      echo "-------------------"
      cat ~/.condarc
      echo "-------------------"
    else
      echo -e "\t ** use image builtin condarc"
    fi
}

set_pip_cache() {
    echo "\t ** set pip cache dir:"
    python3 -m pip config set global.cache-dir ${PIP_CACHE_DIR} || true
    python3 -m pip cache dir || true
}

set_py_and_sw() {
    echo "**** DETECT RUNTIME: python version: ${PYTHON_VERSION}, starwhale version: ${SW_VERSION}"

    echo "-->[Preparing] Set pip config."
    set_pip_config

    echo "-->[Preparing] Use python:${PYTHON_VERSION}."
    set_python_alter
    set_pip_cache

    if [ -z "$SW_VERSION" ]; then
      echo "-->[Preparing] Can't detect starwhale version, use the latest version."
      python3 -m pip install starwhale || exit 1
    else
      # install starwhale for current python
      if [[ $SW_VERSION =~ ^git ]]; then
        echo "-->[Preparing] Install starwhale from git:${SW_VERSION}."
        # SW_VERSION=git+https://github.com/star-whale/starwhale.git@main#subdirectory=client&setup_py=client/setup.py#egg=starwhale
        python3 -m pip install -e "$SW_VERSION" || exit 1
      else
        echo "-->[Preparing] Install starwhale:${SW_VERSION}."
        python3 -m pip install "starwhale==$SW_VERSION" || exit 1
      fi
    fi

}

welcome "$1"
case "$1" in
    set_environment)
        set_py_and_sw
        ;;
    *)
        if [ "${RUNTIME_RESTORED}" != "1" ]; then
          set_py_and_sw
        fi
        if [ "${USE_CUSTOM_CMD}" != "1" ]; then
          sw-docker-entrypoint "$1"
        else
          exec "$@"
        fi
        ;;
esac
