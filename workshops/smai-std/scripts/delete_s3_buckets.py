#!/usr/bin/env python3
"""
S3 Bucket Deletion Script

Deletes one or more S3 buckets, including non-empty buckets. Handles the cases
that normally block `DeleteBucket`:
  - Standard objects
  - Versioned object versions and delete markers (versioned buckets)
  - Incomplete multipart uploads

This is a DESTRUCTIVE and IRREVERSIBLE operation. By default the script runs in
dry-run mode and prints what it would delete. You must pass --execute to actually
delete, and confirm interactively (unless --yes is provided).

Usage:
    # Preview only (safe, default):
    python delete_s3_buckets.py --buckets my-bucket-1 my-bucket-2 --region us-east-1

    # Preview buckets listed in a file (one bucket name per line):
    python delete_s3_buckets.py --buckets-file buckets.txt --region us-east-1

    # Actually delete (prompts for confirmation):
    python delete_s3_buckets.py --buckets my-bucket-1 --region us-east-1 --execute

    # Actually delete without the interactive prompt (use with care, e.g. CI):
    python delete_s3_buckets.py --buckets my-bucket-1 --region us-east-1 --execute --yes

    # Empty the buckets but keep the (now-empty) buckets themselves:
    python delete_s3_buckets.py --buckets my-bucket-1 --region us-east-1 --execute --empty-only
"""

import argparse
import sys
from typing import List

import boto3
from botocore.exceptions import ClientError


class S3BucketDeleter:
    """Empties and (optionally) deletes S3 buckets, including versioned content."""

    def __init__(self, region: str, dry_run: bool = True, empty_only: bool = False):
        """
        Args:
            region: AWS region used for the S3 client.
            dry_run: If True, only report what would be deleted; make no changes.
            empty_only: If True, delete all objects/versions but keep the bucket.
        """
        self.region = region
        self.dry_run = dry_run
        self.empty_only = empty_only
        self.s3 = boto3.client("s3", region_name=region)

    def _log(self, message: str) -> None:
        prefix = "[DRY-RUN] " if self.dry_run else ""
        print(f"{prefix}{message}")

    def bucket_exists(self, bucket: str) -> bool:
        """Return True if the bucket exists and is accessible in this account."""
        try:
            self.s3.head_bucket(Bucket=bucket)
            return True
        except ClientError as e:
            code = e.response["Error"].get("Code", "")
            if code in ("404", "NoSuchBucket"):
                self._log(f"  Bucket '{bucket}' does not exist. Skipping.")
            elif code in ("403", "AccessDenied"):
                self._log(f"  Access denied to bucket '{bucket}'. Skipping.")
            else:
                self._log(f"  Could not access bucket '{bucket}' ({code}). Skipping.")
            return False

    def _delete_object_batch(self, bucket: str, objects: List[dict]) -> None:
        """Delete a batch of up to 1000 objects/versions."""
        if not objects:
            return
        if self.dry_run:
            for obj in objects:
                version = obj.get("VersionId", "null")
                self._log(f"    would delete {obj['Key']} (version: {version})")
            return
        response = self.s3.delete_objects(
            Bucket=bucket,
            Delete={"Objects": objects, "Quiet": True},
        )
        errors = response.get("Errors", [])
        for err in errors:
            print(
                f"    ERROR deleting {err.get('Key')} "
                f"(version: {err.get('VersionId', 'null')}): "
                f"{err.get('Code')} - {err.get('Message')}"
            )

    def empty_bucket(self, bucket: str) -> int:
        """
        Delete every object, object version, and delete marker in the bucket.

        Returns the number of objects/versions processed.
        """
        total = 0

        # Object versions and delete markers (covers both versioned and
        # unversioned buckets; unversioned objects come back with VersionId 'null').
        paginator = self.s3.get_paginator("list_object_versions")
        for page in paginator.paginate(Bucket=bucket):
            batch: List[dict] = []
            for item in page.get("Versions", []) + page.get("DeleteMarkers", []):
                batch.append({"Key": item["Key"], "VersionId": item["VersionId"]})
                total += 1
                if len(batch) == 1000:
                    self._delete_object_batch(bucket, batch)
                    batch = []
            self._delete_object_batch(bucket, batch)

        self._abort_multipart_uploads(bucket)

        if total == 0:
            self._log(f"  Bucket '{bucket}' is already empty.")
        else:
            self._log(f"  Processed {total} object(s)/version(s) in '{bucket}'.")
        return total

    def _abort_multipart_uploads(self, bucket: str) -> None:
        """Abort any incomplete multipart uploads that would block deletion."""
        paginator = self.s3.get_paginator("list_multipart_uploads")
        for page in paginator.paginate(Bucket=bucket):
            for upload in page.get("Uploads", []):
                key, upload_id = upload["Key"], upload["UploadId"]
                if self.dry_run:
                    self._log(f"    would abort multipart upload {key} ({upload_id})")
                    continue
                try:
                    self.s3.abort_multipart_upload(
                        Bucket=bucket, Key=key, UploadId=upload_id
                    )
                except ClientError as e:
                    print(f"    ERROR aborting upload {key} ({upload_id}): {e}")

    def delete_bucket(self, bucket: str) -> bool:
        """
        Empty and delete a single bucket.

        Returns True if the bucket was deleted (or emptied when --empty-only),
        False if it was skipped or failed.
        """
        action = "Emptying" if self.empty_only else "Deleting"
        self._log(f"{action} bucket: {bucket}")

        if not self.bucket_exists(bucket):
            return False

        self.empty_bucket(bucket)

        if self.empty_only:
            return True

        if self.dry_run:
            self._log(f"  would delete bucket '{bucket}'")
            return True

        try:
            self.s3.delete_bucket(Bucket=bucket)
            self._log(f"  Deleted bucket '{bucket}'.")
            return True
        except ClientError as e:
            print(f"  ERROR deleting bucket '{bucket}': {e}")
            return False

    def run(self, buckets: List[str]) -> int:
        """Process all buckets. Returns a process exit code (0 = all succeeded)."""
        succeeded, failed = 0, 0
        for bucket in buckets:
            if self.delete_bucket(bucket):
                succeeded += 1
            else:
                failed += 1
            print()

        verb = "Would process" if self.dry_run else "Processed"
        print(f"{verb}: {succeeded} succeeded, {failed} skipped/failed.")
        return 0 if failed == 0 else 1


def load_bucket_names(args: argparse.Namespace) -> List[str]:
    """Collect bucket names from --buckets and/or --buckets-file."""
    buckets: List[str] = list(args.buckets or [])
    if args.buckets_file:
        with open(args.buckets_file, "r", encoding="utf-8") as f:
            for line in f:
                name = line.strip()
                if name and not name.startswith("#"):
                    buckets.append(name)
    # De-duplicate while preserving order.
    seen = set()
    unique = []
    for b in buckets:
        if b not in seen:
            seen.add(b)
            unique.append(b)
    return unique


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Empty and delete one or more S3 buckets (including non-empty/versioned).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    source = parser.add_argument_group("bucket selection (provide at least one)")
    source.add_argument(
        "--buckets",
        nargs="+",
        metavar="NAME",
        help="One or more bucket names.",
    )
    source.add_argument(
        "--buckets-file",
        metavar="PATH",
        help="Path to a file with one bucket name per line (# comments allowed).",
    )
    parser.add_argument(
        "--region",
        default="us-east-1",
        help="AWS region (default: us-east-1).",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually perform deletions. Without this flag the script only previews (dry-run).",
    )
    parser.add_argument(
        "--empty-only",
        action="store_true",
        help="Delete all objects/versions but keep the (now-empty) buckets.",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Skip the interactive confirmation prompt (only meaningful with --execute).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not args.buckets and not args.buckets_file:
        print("ERROR: provide --buckets and/or --buckets-file.", file=sys.stderr)
        return 2

    buckets = load_bucket_names(args)
    if not buckets:
        print("ERROR: no bucket names resolved from the provided input.", file=sys.stderr)
        return 2

    dry_run = not args.execute

    action = "EMPTY" if args.empty_only else "DELETE"
    print(f"Target region: {args.region}")
    print(f"Mode: {'DRY-RUN (no changes)' if dry_run else 'EXECUTE'}")
    print(f"Action: {action} the following {len(buckets)} bucket(s):")
    for b in buckets:
        print(f"  - {b}")
    print()

    # Confirmation gate for real deletions.
    if not dry_run and not args.yes:
        print(
            "This is DESTRUCTIVE and IRREVERSIBLE. All objects and versions "
            "in these buckets will be permanently deleted."
        )
        confirmation = input('Type "delete" to proceed: ').strip().lower()
        if confirmation != "delete":
            print("Aborted. No changes made.")
            return 1
        print()

    deleter = S3BucketDeleter(
        region=args.region, dry_run=dry_run, empty_only=args.empty_only
    )
    return deleter.run(buckets)


if __name__ == "__main__":
    sys.exit(main())
