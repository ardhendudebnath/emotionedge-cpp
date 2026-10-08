# Test-only TLS material

For the server tests (`tests/server/`): a throwaway CA (`ca.pem`) and a certificate for
`localhost` / `127.0.0.1` signed by it (`server.pem`, key `server.key`), valid until 2126.
The CA's private key was discarded when these were made, so nothing else can be signed with it.

Never use these to serve anything: the key is public. For a real deployment, use a certificate
from your own CA or a public one.
