# Qwen3.8 FP4-Local project guidance

The current project target is NVIDIA's `Qwen3.8-Flash-Next-FP4-Local` on one
DGX Spark. The production service is the user systemd unit
`qwen38-fp4.service`; it runs the guarded `serve.diagnostic` path with a 64K
context, two sequences, a 9 GiB KV budget, MTP 4, FP8 KV and disk PLE.

The agreed safety boundaries are `MemAvailable` 18 GiB, memory PSI full 10%,
and immediate stop on OOM. Do not weaken them. The API is exposed on the LAN
only with the bearer key stored outside the repository in
`~/.config/qwen38/qwen38.env`; never commit that file or print the key.

Operational state is recorded in `README.md` and `serve/README.md`. Do not
stop, restart or change the serving profile without explicit user
authorization, and preserve model caches, diagnostic evidence, and unrelated
services. The normal `serve.launch launch` path remains disabled; the running
service is the guarded systemd path.

Mac OpenCode access and a real reboot have not been verified. Historical
research, including the former self-quantized NVFP4 work, is not part of this
operating path; see `docs/previous-work.md`. Do not delete preserved work
without explicit authorization.

No commit, push, installation, global configuration change, credential access,
provider/model routing, deployment, or pilot follows from this guidance.
