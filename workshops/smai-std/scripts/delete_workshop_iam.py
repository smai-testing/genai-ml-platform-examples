#!/usr/bin/env python3
"""
IAM Orphan Cleanup Script (workshop redeploy helper)

Deletes IAM roles and customer-managed policies left ORPHANED by a failed
CloudFormation deploy of the workshop stack. These carry fixed names
(``<ProjectName>-...``), so a fresh CREATE collides with the leftovers and the
IAMStack fails with "already exists" (surfaced as "Validation failure").

Deleting these is RECOVERABLE: the next successful ``deploy-workshop.sh`` run
recreates them. It is, however, a bulk IAM change on a real account, so this
script is DRY-RUN by default: it lists exactly what it would delete and changes
nothing. Pass --execute to act, and confirm interactively (unless --yes).

Scope is strict: only roles/policies whose name STARTS WITH the given prefix
(default ``bank-marketing-prediction-``) are ever touched. AWS-managed policies
are never deleted (only detached).

Deletion order per role: remove from instance profiles -> detach managed
policies -> delete inline policies -> delete role. Per managed policy: detach
from all attached entities -> delete all non-default versions -> delete policy.

Usage:
    # Preview only (safe, default):
    python delete_workshop_iam.py --prefix bank-marketing-prediction-

    # Actually delete (prompts once):
    python delete_workshop_iam.py --prefix bank-marketing-prediction- --execute

    # Actually delete without the prompt:
    python delete_workshop_iam.py --prefix bank-marketing-prediction- --execute --yes
"""
import argparse
import sys

import boto3
from botocore.exceptions import ClientError


class IamOrphanCleaner:
    def __init__(self, prefix: str, dry_run: bool = True):
        self.prefix = prefix
        self.dry_run = dry_run
        self.iam = boto3.client("iam")

    def _log(self, msg: str) -> None:
        print(f"{'[DRY-RUN] ' if self.dry_run else ''}{msg}")

    # ---- discovery -------------------------------------------------------
    def find_roles(self):
        roles, paginator = [], self.iam.get_paginator("list_roles")
        for page in paginator.paginate():
            roles += [r["RoleName"] for r in page["Roles"]
                      if r["RoleName"].startswith(self.prefix)]
        return sorted(roles)

    def find_policies(self):
        pols, paginator = [], self.iam.get_paginator("list_policies")
        for page in paginator.paginate(Scope="Local"):
            for p in page["Policies"]:
                if p["PolicyName"].startswith(self.prefix) or self.prefix in p["Arn"]:
                    pols.append((p["PolicyName"], p["Arn"]))
        return sorted(pols)

    # ---- role deletion ---------------------------------------------------
    def delete_role(self, role: str) -> None:
        # 1. Remove from instance profiles
        try:
            for ip in self.iam.list_instance_profiles_for_role(RoleName=role)["InstanceProfiles"]:
                name = ip["InstanceProfileName"]
                self._log(f"  role {role}: remove from instance profile {name}")
                if not self.dry_run:
                    self.iam.remove_role_from_instance_profile(
                        InstanceProfileName=name, RoleName=role)
        except ClientError as e:
            self._log(f"  role {role}: list instance profiles failed: {e}")
        # 2. Detach managed policies
        try:
            for ap in self.iam.list_attached_role_policies(RoleName=role)["AttachedPolicies"]:
                self._log(f"  role {role}: detach managed {ap['PolicyName']}")
                if not self.dry_run:
                    self.iam.detach_role_policy(RoleName=role, PolicyArn=ap["PolicyArn"])
        except ClientError as e:
            self._log(f"  role {role}: list attached failed: {e}")
        # 3. Delete inline policies
        try:
            for pn in self.iam.list_role_policies(RoleName=role)["PolicyNames"]:
                self._log(f"  role {role}: delete inline {pn}")
                if not self.dry_run:
                    self.iam.delete_role_policy(RoleName=role, PolicyName=pn)
        except ClientError as e:
            self._log(f"  role {role}: list inline failed: {e}")
        # 4. Delete the role
        self._log(f"DELETE role {role}")
        if not self.dry_run:
            self.iam.delete_role(RoleName=role)

    # ---- policy deletion -------------------------------------------------
    def delete_policy(self, name: str, arn: str) -> None:
        # 1. Detach from all entities
        try:
            ent = self.iam.list_entities_for_policy(PolicyArn=arn)
            for r in ent.get("PolicyRoles", []):
                self._log(f"  policy {name}: detach from role {r['RoleName']}")
                if not self.dry_run:
                    self.iam.detach_role_policy(RoleName=r["RoleName"], PolicyArn=arn)
            for u in ent.get("PolicyUsers", []):
                self._log(f"  policy {name}: detach from user {u['UserName']}")
                if not self.dry_run:
                    self.iam.detach_user_policy(UserName=u["UserName"], PolicyArn=arn)
            for g in ent.get("PolicyGroups", []):
                self._log(f"  policy {name}: detach from group {g['GroupName']}")
                if not self.dry_run:
                    self.iam.detach_group_policy(GroupName=g["GroupName"], PolicyArn=arn)
        except ClientError as e:
            self._log(f"  policy {name}: list entities failed: {e}")
        # 2. Delete non-default versions
        try:
            for v in self.iam.list_policy_versions(PolicyArn=arn)["Versions"]:
                if not v["IsDefaultVersion"]:
                    self._log(f"  policy {name}: delete version {v['VersionId']}")
                    if not self.dry_run:
                        self.iam.delete_policy_version(PolicyArn=arn, VersionId=v["VersionId"])
        except ClientError as e:
            self._log(f"  policy {name}: list versions failed: {e}")
        # 3. Delete the policy
        self._log(f"DELETE policy {name}")
        if not self.dry_run:
            self.iam.delete_policy(PolicyArn=arn)


def main() -> int:
    ap = argparse.ArgumentParser(description="Delete orphaned workshop IAM roles/policies.")
    ap.add_argument("--prefix", default="bank-marketing-prediction-",
                    help="Only roles/policies whose name starts with this are touched.")
    ap.add_argument("--execute", action="store_true", help="Actually delete (default: dry-run).")
    ap.add_argument("--yes", action="store_true", help="Skip the interactive confirmation.")
    args = ap.parse_args()

    cleaner = IamOrphanCleaner(prefix=args.prefix, dry_run=not args.execute)
    roles = cleaner.find_roles()
    policies = cleaner.find_policies()

    print(f"Prefix: {args.prefix!r}")
    print(f"Matched roles ({len(roles)}):")
    for r in roles:
        print(f"  - {r}")
    print(f"Matched customer-managed policies ({len(policies)}):")
    for n, _ in policies:
        print(f"  - {n}")

    if not roles and not policies:
        print("Nothing to clean up.")
        return 0

    if args.execute and not args.yes:
        resp = input(f"\nDelete {len(roles)} roles and {len(policies)} policies? Type 'yes': ")
        if resp.strip().lower() != "yes":
            print("Aborted.")
            return 1

    print()
    # Delete policies first? No — detach happens per-role and per-policy, order
    # is safe either way because each handler detaches its own dependencies.
    for r in roles:
        cleaner.delete_role(r)
    for n, arn in policies:
        cleaner.delete_policy(n, arn)

    print("\nDone." if args.execute else "\nDry-run complete — no changes made. Re-run with --execute to delete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
