# Security

## Report a problem

Do not open a public issue for a security problem. Use **Report a vulnerability** on the Security tab of the repository. Do not include private profiles.

## Security model

- The server has no authentication and no TLS. It listens on `127.0.0.1` by default. With `--host`, any computer that can connect can read all profiles in the log directory.
- With `--logdir`, a request can read only the files in the log directory. A path outside the log directory gets a 400 response. The directory walk does not follow symbolic links to directories.
- Without `--logdir`, a `session_path` or `run_path` request can open any directory that the server can read. Use `--logdir` on a shared machine.
- `/capture_profile` connects to the address in the request and writes the captured profile to the log directory.
