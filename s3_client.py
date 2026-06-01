"""
Standalone S3 client for the floor-plan-generation app.

Replaces the previous core.utils.s3_helper.s3 import from the Mesaky backend.

AWS credentials are picked up from the environment in priority order:
  1. Explicit env vars: AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY + AWS_REGION
  2. IAM instance role / ECS task role (boto3 default credential chain)

Configure in .env:
  AWS_ACCESS_KEY_ID=...
  AWS_SECRET_ACCESS_KEY=...
  AWS_REGION=ap-south-1          # or AWS_S3_REGION_NAME
"""

import os

import boto3


def get_s3_client():
    """Return a boto3 S3 client using credentials from the environment."""
    region = (
        os.getenv("AWS_REGION")
        or os.getenv("AWS_S3_REGION_NAME")
        or os.getenv("AWS_S3_REGION_NAME_new")
    )

    kwargs = {}
    if region:
        kwargs["region_name"] = region

    access_key = os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    if access_key and secret_key:
        kwargs["aws_access_key_id"] = access_key
        kwargs["aws_secret_access_key"] = secret_key
        session_token = os.getenv("AWS_SESSION_TOKEN")
        if session_token:
            kwargs["aws_session_token"] = session_token

    return boto3.client("s3", **kwargs)


# Module-level singleton — created lazily on first import so that missing
# credentials at import time don't crash the whole app.
_s3 = None


def s3():
    """Return the shared S3 client, creating it on first call."""
    global _s3
    if _s3 is None:
        _s3 = get_s3_client()
    return _s3
