# v4 migration rollover image: upstream dev-dspark + fleet ray pin + the
# dspark_port_series (fp4_w4a16_design.md Phase 4/6, GPU-validated
# 2026-07-17 on node53). Build context: scripts/dsv4/patches/dspark_port_series/.
ARG DSPARK_BASE_IMAGE=m.daocloud.io/docker.io/lmsysorg/sglang:dev-dspark
FROM ${DSPARK_BASE_IMAGE}
COPY *.patch /opt/dspark_port_series/
RUN cd /sgl-workspace/sglang \
    && test "$(git rev-parse HEAD)" = "692c5f7d532f129424b57961c262bbd253b411dc" \
    && for p in /opt/dspark_port_series/00*.patch; do \
         git apply --check "$p" && git apply "$p" || exit 1; \
       done \
    && test "$(git diff --binary | sha256sum | cut -d' ' -f1)" = \
       "d6261fe413b157ba4f06073a35cdce85e834cc8e5bd4efdb787345e1d091f2be" \
    && grep -q 'self.lora_enabled' python/sglang/srt/model_executor/model_runner.py \
    && python -c 'import sglang.srt.server_args'
# fleet ray pin (csl_slime_0614 fleet containers run ray 2.55.1) + slime
# rollout-actor deps missing from the sglang base (deep-import probe
# 2026-07-17: wandb via slime.utils, pylatexenc via rm_hub math_utils).
RUN pip install --no-cache-dir "ray[default]==2.55.1" wandb pylatexenc
