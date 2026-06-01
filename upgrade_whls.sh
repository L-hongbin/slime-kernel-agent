export PIP_BREAK_SYSTEM_PACKAGES=1
bash /ms/FM/lihongbin/code/scripts/set_pip_source.sh
pip install --no-deps -e .
pip install ray==2.53.0
pip install --force-reinstall /data/FM/lhb/whls/torch_memory_saver-0.0.9.post1-cp312-cp312-linux_x86_64.whl