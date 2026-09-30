"""Python vaccines for the showcase. Referenced from vaccines/bobs.vip_accounts.yaml."""

from __future__ import annotations

from immune.vaccines import VaccineContext


def touches_vip_account(context: VaccineContext) -> str | None:
    account = str(context.arguments.get("account", ""))
    return f"account {account} is a VIP account" if account.startswith("vip-") else None
