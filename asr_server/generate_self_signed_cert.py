"""
generate_self_signed_cert.py

Creates a self-signed TLS certificate + private key for local testing of
asr_server.py. Self-signed is fine for development on your own network;
for anything beyond your bench, use a properly issued certificate
(e.g. from Let's Encrypt, which is free and open) instead.

Usage:
    python generate_self_signed_cert.py [--hostname YOUR_SERVER_IP_OR_NAME]
"""

import argparse
import datetime
import ipaddress
import os

from cryptography import x509
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.x509.oid import NameOID


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hostname", type=str, default="localhost",
                         help="The hostname or IP your ESP32 will connect to")
    parser.add_argument("--out_dir", type=str, default="certs")
    parser.add_argument("--days", type=int, default=365)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)

    subject = issuer = x509.Name([
        x509.NameAttribute(NameOID.COMMON_NAME, args.hostname),
    ])

    san_entries = [x509.DNSName(args.hostname)]
    try:
        san_entries.append(x509.IPAddress(ipaddress.ip_address(args.hostname)))
    except ValueError:
        pass  # hostname isn't a bare IP, DNSName entry above is enough

    now = datetime.datetime.now(datetime.timezone.utc)
    cert = (
        x509.CertificateBuilder()
        .subject_name(subject)
        .issuer_name(issuer)
        .public_key(key.public_key())
        .serial_number(x509.random_serial_number())
        .not_valid_before(now)
        .not_valid_after(now + datetime.timedelta(days=args.days))
        .add_extension(x509.SubjectAlternativeName(san_entries), critical=False)
        .sign(key, hashes.SHA256())
    )

    key_path = os.path.join(args.out_dir, "server.key")
    cert_path = os.path.join(args.out_dir, "server.crt")

    with open(key_path, "wb") as f:
        f.write(key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.TraditionalOpenSSL,
            encryption_algorithm=serialization.NoEncryption(),
        ))

    with open(cert_path, "wb") as f:
        f.write(cert.public_bytes(serialization.Encoding.PEM))

    print(f"Wrote {key_path} and {cert_path} (valid {args.days} days, "
          f"hostname={args.hostname})")
    print("\nFor the ESP32 to verify this server (recommended over "
          "setInsecure()), copy the contents of server.crt into "
          "esp32_firmware/kws_firmware/certs.h as ASR_SERVER_CA_CERT.")


if __name__ == "__main__":
    main()
