from __future__ import annotations

from immune.vaccines import VaccineContext


def touches_vip_account(context: VaccineContext) -> str | None:
    account = str(context.arguments.get("account", ""))
    return f"account {account} is a VIP account" if account.startswith("vip-") else None


def always_fails(context: VaccineContext) -> bool:
    raise RuntimeError("broken vaccine")
