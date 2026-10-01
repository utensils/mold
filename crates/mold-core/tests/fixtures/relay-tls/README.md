This self-signed certificate and synthetic private key are fixtures for the
staged-upload TLS integration test. They contain no deployment credentials.
The test trusts this certificate explicitly and resolves its synthetic S3
hostname to a temporary loopback listener. Do not use this key in a deployment.

The certificate expires in September 2036; renew the pair before then.
