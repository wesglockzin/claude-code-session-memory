---
name: reference_vpn_cert_bundle
description: "Corporate VPN does TLS inspection; tools need the internal CA bundle"
metadata:
  type: reference
---

The corporate VPN intercepts TLS. Python tools must set
REQUESTS_CA_BUNDLE to the internal bundle at ~/.corp-ca-bundle.pem
or requests fail with certificate errors that look like outages.
Node tools need NODE_EXTRA_CA_CERTS pointed at the same file.
