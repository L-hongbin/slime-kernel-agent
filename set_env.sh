if [[ ! -d "/nfs/FM/chenshuailin/checkpoints" ]]; then
    mkdir -p /nfs/FM/chenshuailin/
    ln -s /ms/FM/checkpoints /nfs/FM/chenshuailin/checkpoints
fi


cd $(dirname $0)
pip install -e . --no-deps --break-system-packages
