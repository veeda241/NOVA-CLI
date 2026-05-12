from __future__ import annotations

import calendar
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo, available_timezones


class DateTimeTool:
    def get_current(self, timezone_name: str = "auto") -> dict:
        local_tz = datetime.now().astimezone().tzinfo if timezone_name == "auto" else ZoneInfo(timezone_name)
        now = datetime.now(tz=local_tz)
        utc_now = datetime.now(timezone.utc)
        return {
            "datetime": now.isoformat(),
            "date": now.strftime("%Y-%m-%d"),
            "time": now.strftime("%H:%M:%S"),
            "time_12h": now.strftime("%I:%M %p"),
            "day_name": now.strftime("%A"),
            "day_number": now.day,
            "month_name": now.strftime("%B"),
            "month_number": now.month,
            "year": now.year,
            "week_number": now.isocalendar().week,
            "day_of_year": now.timetuple().tm_yday,
            "is_weekend": now.weekday() >= 5,
            "is_leap_year": calendar.isleap(now.year),
            "quarter": (now.month - 1) // 3 + 1,
            "timezone": str(local_tz),
            "utc_offset": now.strftime("%z"),
            "utc_time": utc_now.strftime("%H:%M:%S"),
            "unix_timestamp": int(now.timestamp()),
            "days_until_year_end": (date(now.year, 12, 31) - now.date()).days,
            "days_since_year_start": now.timetuple().tm_yday - 1,
        }

    def get_time_in_timezone(self, timezone_name: str) -> dict:
        try:
            tz = ZoneInfo(timezone_name)
            now = datetime.now(tz=tz)
            return {
                "timezone": timezone_name,
                "datetime": now.isoformat(),
                "date": now.strftime("%Y-%m-%d"),
                "time": now.strftime("%H:%M:%S"),
                "time_12h": now.strftime("%I:%M %p"),
                "day_name": now.strftime("%A"),
                "utc_offset": now.strftime("%z"),
            }
        except Exception as exc:
            return {"error": str(exc)}

    def get_time_difference(self, from_date: str, to_date: str | None = None) -> dict:
        try:
            from_dt = datetime.fromisoformat(from_date)
            to_dt = datetime.fromisoformat(to_date) if to_date else datetime.now(from_dt.tzinfo)
            diff = abs(to_dt - from_dt)
            total_seconds = int(diff.total_seconds())
            return {
                "days": diff.days,
                "hours": diff.seconds // 3600,
                "minutes": (diff.seconds % 3600) // 60,
                "seconds": diff.seconds % 60,
                "total_seconds": total_seconds,
                "total_minutes": total_seconds // 60,
                "total_hours": total_seconds // 3600,
                "total_weeks": diff.days // 7,
                "human": self._human_timedelta(diff),
            }
        except Exception as exc:
            return {"error": str(exc)}

    def get_calendar(self, month: int | None = None, year: int | None = None) -> dict:
        now = datetime.now()
        selected_month = month or now.month
        selected_year = year or now.year
        first_weekday, days_in_month = calendar.monthrange(selected_year, selected_month)
        return {
            "month": calendar.month_name[selected_month],
            "year": selected_year,
            "days_in_month": days_in_month,
            "first_weekday": calendar.day_name[first_weekday],
            "calendar_grid": calendar.monthcalendar(selected_year, selected_month),
            "today": now.day if selected_month == now.month and selected_year == now.year else None,
        }

    def calculate_date(self, base_date: str, days: int = 0, weeks: int = 0, months: int = 0) -> dict:
        try:
            from dateutil.relativedelta import relativedelta

            base = datetime.fromisoformat(base_date)
            result = base + relativedelta(days=days, weeks=weeks, months=months)
            return {
                "result_date": result.strftime("%Y-%m-%d"),
                "result_day": result.strftime("%A"),
                "result_formatted": result.strftime("%B %d, %Y"),
            }
        except Exception as exc:
            return {"error": str(exc)}

    def get_all_timezones(self) -> list[str]:
        return sorted(available_timezones())

    def _human_timedelta(self, delta: timedelta) -> str:
        days = delta.days
        if days == 0:
            return "today"
        if days == 1:
            return "1 day"
        if days < 7:
            return f"{days} days"
        if days < 30:
            return f"{days // 7} weeks"
        if days < 365:
            return f"{days // 30} months"
        return f"{days // 365} years"


datetime_tool = DateTimeTool()
