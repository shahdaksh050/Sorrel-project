# Sandbox image

The image the Docker backend (`SANDBOX_BACKEND=docker`) runs LLM-authored code in.

## Build

From the repo root — rebuild after any change under `src/`:

```
docker build -t dsa-sandbox:latest .
```

`requirements-sandbox.txt` holds only what the worker imports, pinned to `requirements.lock`.
Pin the base image by digest for reproducible builds (see the comment in `Dockerfile`).

## Seccomp profile

`seccomp-sandbox.json` is an allowlist: anything not listed fails with `ENOSYS`. Compared with
Docker's default profile it additionally denies:

- networking — `socket`/`socketpair` only for `AF_UNIX`; `connect`, `bind`, `listen`, `accept`,
  `sendto`, `recvfrom` and the rest are absent
- `ptrace`, `process_vm_readv`, `process_vm_writev`
- `mount`, `umount2`, `pivot_root`, `unshare`, `setns`, and `clone` with namespace flags
- `keyctl`, `add_key`, `request_key`, `bpf`, `perf_event_open`, `personality`

`DockerSandbox` should add this to its `docker run` command, next to `no-new-privileges`, with the path
resolved to an absolute one (the Streamlit process's working directory is not guaranteed to be the
repo root):

```
--security-opt seccomp=/abs/path/to/docker/seccomp-sandbox.json
```

e.g. `"--security-opt", f"seccomp={Path(__file__).resolve().parents[2] / 'docker' / 'seccomp-sandbox.json'}"`
in `src/core/sandbox.py`.

If a library starts failing inside the container with `Function not implemented`, a syscall it needs
is missing from the allowlist — add it to the matching group, not a blanket allow.
