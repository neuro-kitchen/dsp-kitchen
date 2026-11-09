---
title: Smoke test
description: "Check an image end to end with nothing on the host but Podman."
sidebar:
  order: 8
---

`deploy/smoke-test.sh` uses the image as both server and client:

```sh
deploy/smoke-test.sh                    # target from the host: nvidia, mesa or cpu
deploy/smoke-test.sh --target cpu --no-build
```

It goes through these steps, printing PASS or FAIL for each:

1. Build the image.
2. `doctor`: the target's runtime is usable.
3. Write a synthetic recording.
4. Serve it, hardened.
5. Enroll a read-only client and a control client.
6. Receive the signal.
7. Check refusals: no identity, a sort from the read-only client, fetching the server's key.
8. Run a sort on the server.
9. Fetch the result and check it is identical.
10. Check the audit log.
11. Stop.

The script stops at the first failure, keeping its working folder for a look, and always removes
its container.
