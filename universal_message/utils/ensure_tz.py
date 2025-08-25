import zoneinfo


def ensure_tz(tz: zoneinfo.ZoneInfo | str | None) -> zoneinfo.ZoneInfo:
    """Ensure timezone is ZoneInfo object."""
    if tz is None:
        return zoneinfo.ZoneInfo("UTC")
    if not isinstance(tz, zoneinfo.ZoneInfo):
        return zoneinfo.ZoneInfo(tz)
    return tz
