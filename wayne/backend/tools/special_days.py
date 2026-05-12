from __future__ import annotations

import calendar
from datetime import date, datetime, timedelta


class SpecialDaysEngine:
    def __init__(self) -> None:
        self.fixed_days = self._build_fixed_days()
        self.dynamic_days = {
            "mothers_day": {"name": "Mother's Day", "countries": ["US", "IN", "global"]},
            "fathers_day": {"name": "Father's Day", "countries": ["US", "IN", "global"]},
            "thanksgiving": {"name": "Thanksgiving Day", "countries": ["US"]},
            "easter": {"name": "Easter Sunday", "countries": ["global"]},
            "good_friday": {"name": "Good Friday", "countries": ["global", "IN", "UK"]},
        }

    def _build_fixed_days(self) -> dict[str, dict]:
        rows = [
            ("01-01", "New Year's Day", "global_holiday", ["global"], "First day of the Gregorian calendar year"),
            ("01-14", "Makar Sankranti / Pongal", "indian_festival", ["IN"], "Indian harvest festival"),
            ("01-15", "Army Day", "indian_national", ["IN"], "Indian Army Day"),
            ("01-23", "Netaji Subhas Chandra Bose Jayanti", "indian_national", ["IN"], "Birthday of Netaji Subhas Chandra Bose"),
            ("01-26", "Republic Day", "indian_national_holiday", ["IN"], "India's constitution came into effect in 1950"),
            ("02-04", "World Cancer Day", "global_observance", ["global"], "Cancer awareness day"),
            ("02-14", "Valentine's Day", "global_observance", ["global"], "Celebration of love and affection"),
            ("02-21", "International Mother Language Day", "global_observance", ["global"], "Promotes linguistic diversity"),
            ("02-28", "National Science Day", "indian_national", ["IN"], "Commemorates discovery of the Raman Effect"),
            ("03-08", "International Women's Day", "global_observance", ["global"], "Celebrates women's achievements"),
            ("03-14", "Pi Day", "global_observance", ["global"], "Celebrates pi, 3.14"),
            ("03-20", "International Day of Happiness", "global_observance", ["global"], "UN day of happiness"),
            ("03-21", "World Poetry Day", "global_observance", ["global"], "UNESCO day for poetry"),
            ("03-22", "World Water Day", "global_observance", ["global"], "Freshwater awareness day"),
            ("04-01", "April Fools' Day", "cultural", ["global"], "Day of pranks and jokes"),
            ("04-07", "World Health Day", "global_observance", ["global"], "WHO annual health awareness day"),
            ("04-13", "Baisakhi / Vaisakhi", "indian_festival", ["IN"], "Punjabi harvest festival and Sikh New Year"),
            ("04-14", "Dr. B.R. Ambedkar Jayanti", "indian_national_holiday", ["IN"], "Birthday of Dr. B.R. Ambedkar"),
            ("04-22", "Earth Day", "global_observance", ["global"], "Environmental protection awareness day"),
            ("04-23", "World Book Day", "global_observance", ["global"], "UNESCO day for reading and publishing"),
            ("05-01", "International Workers' Day / Labour Day", "global_holiday", ["global"], "Celebrates workers worldwide"),
            ("05-03", "World Press Freedom Day", "global_observance", ["global"], "Press freedom awareness day"),
            ("05-04", "Star Wars Day", "cultural", ["global"], "May the Fourth be with you"),
            ("05-08", "World Red Cross Day", "global_observance", ["global"], "Honors Red Cross humanitarian work"),
            ("05-12", "International Nurses Day", "global_observance", ["global"], "Honors nurses on Florence Nightingale's birthday"),
            ("05-15", "International Day of Families", "global_observance", ["global"], "UN day for families"),
            ("05-21", "National Anti-Terrorism Day", "indian_national", ["IN"], "Observed in India on Rajiv Gandhi's death anniversary"),
            ("05-31", "World No Tobacco Day", "global_observance", ["global"], "WHO tobacco risk awareness day"),
            ("06-05", "World Environment Day", "global_observance", ["global"], "UN environmental awareness day"),
            ("06-08", "World Ocean Day", "global_observance", ["global"], "Celebrates the world's oceans"),
            ("06-14", "World Blood Donor Day", "global_observance", ["global"], "Blood donation awareness day"),
            ("06-21", "International Yoga Day / World Music Day", "global_observance", ["global", "IN"], "UN day promoting yoga practice"),
            ("07-01", "National Doctor's Day", "indian_national", ["IN"], "Honors Dr. Bidhan Chandra Roy"),
            ("07-04", "Independence Day", "us_holiday", ["US"], "United States Independence Day"),
            ("07-11", "World Population Day", "global_observance", ["global"], "Population issues awareness day"),
            ("07-26", "Kargil Vijay Diwas", "indian_national", ["IN"], "India's Kargil War victory day"),
            ("08-12", "International Youth Day", "global_observance", ["global"], "UN day for youth issues"),
            ("08-15", "Independence Day", "indian_national_holiday", ["IN"], "India's independence from British rule in 1947"),
            ("08-29", "National Sports Day", "indian_national", ["IN"], "Birthday of Major Dhyan Chand"),
            ("09-05", "Teachers' Day", "indian_national", ["IN"], "Birthday of Dr. Sarvepalli Radhakrishnan"),
            ("09-08", "International Literacy Day", "global_observance", ["global"], "UNESCO literacy awareness day"),
            ("09-14", "Hindi Diwas", "indian_national", ["IN"], "Celebrates Hindi as an official language of India"),
            ("09-21", "International Day of Peace", "global_observance", ["global"], "UN day for world peace"),
            ("09-27", "World Tourism Day", "global_observance", ["global"], "UNWTO tourism day"),
            ("10-02", "Gandhi Jayanti / International Day of Non-Violence", "indian_national_holiday", ["IN", "global"], "Mahatma Gandhi's birthday"),
            ("10-05", "World Teachers' Day", "global_observance", ["global"], "UNESCO day celebrating teachers"),
            ("10-08", "Indian Air Force Day", "indian_national", ["IN"], "Indian Air Force founding anniversary"),
            ("10-10", "World Mental Health Day", "global_observance", ["global"], "Mental health awareness day"),
            ("10-16", "World Food Day", "global_observance", ["global"], "FAO founding anniversary"),
            ("10-24", "United Nations Day", "global_observance", ["global"], "UN Charter anniversary"),
            ("10-31", "Halloween / National Unity Day", "cultural", ["global", "IN"], "Halloween and India's Rashtriya Ekta Diwas"),
            ("11-11", "Remembrance Day / Veterans Day", "global_observance", ["global", "US", "UK"], "Honors war veterans"),
            ("11-14", "Children's Day / World Diabetes Day", "indian_national", ["IN", "global"], "Nehru's birthday and diabetes awareness day"),
            ("11-19", "International Men's Day / World Toilet Day", "global_observance", ["global"], "Men's day and sanitation awareness day"),
            ("11-20", "Universal Children's Day", "global_observance", ["global"], "UN day for children's welfare"),
            ("12-01", "World AIDS Day", "global_observance", ["global"], "HIV/AIDS awareness day"),
            ("12-02", "National Pollution Control Day", "indian_national", ["IN"], "Remembers Bhopal Gas Tragedy victims"),
            ("12-03", "International Day of Persons with Disabilities", "global_observance", ["global"], "Disability rights awareness day"),
            ("12-04", "Indian Navy Day", "indian_national", ["IN"], "Indian Navy Day"),
            ("12-10", "Human Rights Day", "global_observance", ["global"], "UDHR adoption anniversary"),
            ("12-16", "Vijay Diwas", "indian_national", ["IN"], "India's 1971 victory day"),
            ("12-22", "National Mathematics Day", "indian_national", ["IN"], "Birthday of Srinivasa Ramanujan"),
            ("12-23", "Kisan Diwas / National Farmers Day", "indian_national", ["IN"], "Birthday of Chaudhary Charan Singh"),
            ("12-24", "Christmas Eve", "global_observance", ["global"], "Evening before Christmas"),
            ("12-25", "Christmas Day", "global_holiday", ["global"], "Christian celebration of the birth of Jesus Christ"),
            ("12-31", "New Year's Eve", "global_observance", ["global"], "Last day of the Gregorian year"),
        ]
        return {key: {"name": name, "type": kind, "countries": countries, "description": desc} for key, name, kind, countries, desc in rows}

    def get_today_specials(self, country: str = "IN") -> dict:
        now = datetime.now()
        return self.get_day_info(now.strftime("%Y-%m-%d"), country) | {"today": now.strftime("%B %d, %Y")}

    def get_upcoming(self, days_ahead: int = 30, country: str = "IN") -> list[dict]:
        now = datetime.now()
        upcoming: list[dict] = []
        for offset in range(1, days_ahead + 1):
            check = now + timedelta(days=offset)
            for item in self._events_for_date(check.date(), country):
                upcoming.append(
                    item
                    | {
                        "date": check.strftime("%Y-%m-%d"),
                        "days_away": offset,
                        "formatted_date": check.strftime("%B %d, %A"),
                    }
                )
        return upcoming

    def get_day_info(self, date_str: str, country: str = "IN") -> dict:
        try:
            check = datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            return {"error": "Invalid date format. Use YYYY-MM-DD"}
        events = self._events_for_date(check.date(), country)
        return {
            "date": date_str,
            "day_name": check.strftime("%A"),
            "formatted": check.strftime("%B %d, %Y"),
            "special_days": events,
            "has_special_day": bool(events),
            "is_weekend": check.weekday() >= 5,
        }

    def _events_for_date(self, check: date, country: str) -> list[dict]:
        events: list[dict] = []
        key = check.strftime("%m-%d")
        if key in self.fixed_days and self._matches_country(self.fixed_days[key], country):
            events.append(self.fixed_days[key] | {"date": check.isoformat(), "days_away": 0})
        for event_id, info in self.dynamic_days.items():
            calc_date = self._calculate_dynamic(event_id, check.year)
            if calc_date == check and ("global" in info["countries"] or country in info["countries"]):
                events.append(
                    {
                        "name": info["name"],
                        "type": "dynamic_observance",
                        "countries": info["countries"],
                        "description": f"{info['name']} for {check.year}",
                        "date": check.isoformat(),
                        "days_away": 0,
                    }
                )
        return events

    def _matches_country(self, day: dict, country: str) -> bool:
        return "global" in day.get("countries", []) or country in day.get("countries", [])

    def _calculate_dynamic(self, day_id: str, year: int) -> date | None:
        if day_id == "mothers_day":
            sundays = [week[6] for week in calendar.monthcalendar(year, 5) if week[6]]
            return date(year, 5, sundays[1])
        if day_id == "fathers_day":
            sundays = [week[6] for week in calendar.monthcalendar(year, 6) if week[6]]
            return date(year, 6, sundays[2])
        if day_id == "thanksgiving":
            thursdays = [week[3] for week in calendar.monthcalendar(year, 11) if week[3]]
            return date(year, 11, thursdays[3])
        if day_id in {"easter", "good_friday"}:
            easter = self._easter_date(year)
            return easter - timedelta(days=2) if day_id == "good_friday" else easter
        return None

    def _easter_date(self, year: int) -> date:
        a = year % 19
        b = year // 100
        c = year % 100
        d = b // 4
        e = b % 4
        f = (b + 8) // 25
        g = (b - f + 1) // 3
        h = (19 * a + b - d - g + 15) % 30
        i = c // 4
        k = c % 4
        l = (32 + 2 * e + 2 * i - h - k) % 7
        m = (a + 11 * h + 22 * l) // 451
        month = (h + l - 7 * m + 114) // 31
        day = ((h + l - 7 * m + 114) % 31) + 1
        return date(year, month, day)


special_days_engine = SpecialDaysEngine()
