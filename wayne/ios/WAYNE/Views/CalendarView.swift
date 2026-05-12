import SwiftUI

struct CalendarView: View {
    @State private var events: [CalendarEvent] = []
    @State private var dateInfo: DateInfo = .fallback
    @State private var todaySpecials: [SpecialDay] = []
    @State private var upcomingSpecials: [SpecialDay] = []
    @State private var ticker = Date()
    private let colors: [Color] = [.wayneAccent, .wayneAmber, .wayneGreen]
    private let timer = Timer.publish(every: 1, on: .main, in: .common).autoconnect()

    var body: some View {
        List {
            Section("TODAY") {
                VStack(alignment: .leading, spacing: 8) {
                    Text("\(dateInfo.dayName), \(dateInfo.monthName) \(dateInfo.dayNumber)")
                        .font(.title3.monospaced().bold())
                        .foregroundStyle(Color.wayneAccent)
                    Text(DateFormatter.localizedString(from: ticker, dateStyle: .none, timeStyle: .medium))
                        .font(.body.monospaced())
                        .foregroundStyle(Color.wayneAmber)
                    Text("Week \(dateInfo.weekNumber) | Day \(dateInfo.dayOfYear) of \(dateInfo.year)")
                        .font(.caption.monospaced())
                        .foregroundStyle(Color.wayneAccent.opacity(0.65))
                    if !todaySpecials.isEmpty {
                        Text("Today is \(todaySpecials.map(\.name).joined(separator: ", "))")
                            .font(.caption.monospaced())
                            .foregroundStyle(Color.wayneAmber)
                            .padding(8)
                            .background(Color.wayneAmber.opacity(0.12))
                    }
                }
                .listRowBackground(Color.wayneSurface)
            }

            if !upcomingSpecials.isEmpty {
                Section("UPCOMING SPECIAL DAYS") {
                    ForEach(upcomingSpecials.prefix(5)) { day in
                        VStack(alignment: .leading, spacing: 3) {
                            Text(day.name)
                                .font(.body.monospaced())
                                .foregroundStyle(.white)
                            Text("\(day.formattedDate ?? day.date ?? "")\(day.daysAway.map { " | in \($0) days" } ?? "")")
                                .font(.caption.monospaced())
                                .foregroundStyle(Color.wayneAccent.opacity(0.7))
                        }
                        .listRowBackground(Color.wayneSurface)
                    }
                }
            }

            Section("EVENTS") {
            if events.isEmpty {
                Text("No events today")
                    .font(.body.monospaced())
                    .foregroundStyle(Color.wayneAccent.opacity(0.7))
                    .listRowBackground(Color.wayneBackground)
            }
            ForEach(Array(events.enumerated()), id: \.offset) { index, event in
                HStack {
                    Rectangle().fill(colors[index % colors.count]).frame(width: 3)
                    Text(String(event.start.dropFirst(11).prefix(5)))
                        .font(.caption.monospaced())
                        .foregroundStyle(Color.wayneAccent)
                    Text(event.title)
                        .font(.body.monospaced())
                        .foregroundStyle(.white)
                }
                .listRowBackground(Color.wayneSurface)
            }
            }
        }
        .scrollContentBackground(.hidden)
        .background(Color.wayneBackground.ignoresSafeArea())
        .navigationTitle("Calendar")
        .onReceive(timer) { ticker = $0 }
        .task { await load() }
        .refreshable { await load() }
    }

    private func load() async {
        async let loadedEvents = WAYNEService.shared.getEvents()
        async let loadedDate = WAYNEService.shared.getDateInfo()
        async let loadedToday = WAYNEService.shared.getTodaySpecialDays()
        async let loadedUpcoming = WAYNEService.shared.getUpcomingSpecialDays()
        let today = await loadedToday
        events = await loadedEvents
        dateInfo = await loadedDate
        todaySpecials = today.specialDays
        upcomingSpecials = await loadedUpcoming
    }
}
