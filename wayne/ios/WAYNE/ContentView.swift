import SwiftUI

struct ContentView: View {
    var body: some View {
        NavigationStack {
            TabView {
                ChatView()
                    .tabItem { Label("Chat", systemImage: "message.fill") }
                VoiceView()
                    .tabItem { Label("Voice", systemImage: "waveform") }
                DeviceControlView()
                    .tabItem { Label("Devices", systemImage: "laptopcomputer") }
                TaskView()
                    .tabItem { Label("Tasks", systemImage: "checklist") }
                CalendarView()
                    .tabItem { Label("Calendar", systemImage: "calendar") }
                FileView()
                    .tabItem { Label("Files", systemImage: "folder.fill") }
                PCManagerView()
                    .tabItem { Label("PC", systemImage: "cpu") }
                LearningView()
                    .tabItem { Label("Learning", systemImage: "brain.head.profile") }
            }
            .tint(Color.wayneAccent)
            .toolbarBackground(Color.wayneSurface, for: .tabBar)
            .navigationTitle("W.A.Y.N.E")
        }
    }
}
