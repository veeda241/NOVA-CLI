import SwiftUI

struct CalendarView: View {
    @State private var events: [CalendarEvent] = []
    private let colors: [Color] = [.wayneAccent, .wayneAmber, .wayneGreen]

    var body: some View {
        List {
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
        .scrollContentBackground(.hidden)
        .background(Color.wayneBackground.ignoresSafeArea())
        .navigationTitle("Calendar")
        .task { events = await WAYNEService.shared.getEvents() }
        .refreshable { events = await WAYNEService.shared.getEvents() }
    }
}
