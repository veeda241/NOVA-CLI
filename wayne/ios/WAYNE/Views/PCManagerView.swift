import SwiftUI

struct PCManagerView: View {
    @State private var status = SystemStatus.empty
    @State private var processes: [ProcessItem] = []
    @State private var result = ""
    @State private var loading = false

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 14) {
                    statusCard
                    actionGrid
                    processList
                }
                .padding()
            }
            .background(Color.wayneBackground.ignoresSafeArea())
            .navigationTitle("PC Manager")
            .task { await refresh() }
            .refreshable { await refresh() }
        }
    }

    private var statusCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("System Health")
                .font(.headline.monospaced())
                .foregroundStyle(Color.wayneAccent)
            metric("CPU", value: status.cpuPercent / 100, label: "\(Int(status.cpuPercent))%")
            metric("Memory", value: (status.memory?.percent ?? 0) / 100, label: "\(Int(status.memory?.percent ?? 0))%")
            metric("Disk", value: (status.diskPercent ?? 0) / 100, label: "\(Int(status.diskPercent ?? 0))%")
            if let battery = status.batteryPercent {
                metric("Battery", value: battery / 100, label: "\(Int(battery))%")
            }
            if !result.isEmpty {
                Text(result)
                    .font(.caption.monospaced())
                    .foregroundStyle(Color.wayneGreen)
            }
        }
        .padding()
        .background(Color.waynePanel)
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }

    private var actionGrid: some View {
        LazyVGrid(columns: [GridItem(.flexible()), GridItem(.flexible())], spacing: 10) {
            managerButton("Clear Cache", icon: "trash") { await run { await WAYNEService.shared.clearCache() } }
            managerButton("Optimize RAM", icon: "memorychip") { await run { await WAYNEService.shared.optimizeMemory() } }
            managerButton("Refresh", icon: "arrow.clockwise") { await refresh() }
            managerButton("Processes", icon: "list.bullet.rectangle") { processes = await WAYNEService.shared.getProcesses() }
        }
    }

    private var processList: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Top Processes")
                .font(.headline.monospaced())
                .foregroundStyle(Color.wayneAccent)
            ForEach(processes.prefix(12)) { process in
                HStack {
                    VStack(alignment: .leading) {
                        Text(process.name)
                            .foregroundStyle(.white)
                            .font(.body.monospaced())
                        Text("PID \(process.pid) | CPU \(Int(process.cpuPercent ?? 0))% | RAM \(String(format: "%.1f", process.memoryPercent ?? 0))%")
                            .font(.caption2.monospaced())
                            .foregroundStyle(Color.wayneMuted)
                    }
                    Spacer()
                    Button(role: .destructive) {
                        Task {
                            await WAYNEService.shared.killProcess(name: process.name)
                            await refresh()
                        }
                    } label: {
                        Image(systemName: "xmark.circle")
                    }
                }
                .padding(10)
                .background(Color.wayneSurface)
                .clipShape(RoundedRectangle(cornerRadius: 6))
            }
        }
        .padding()
        .background(Color.waynePanel)
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }

    private func metric(_ title: String, value: Double, label: String) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack {
                Text(title).foregroundStyle(Color.wayneMuted)
                Spacer()
                Text(label).foregroundStyle(Color.wayneAmber)
            }
            .font(.caption.monospaced())
            ProgressView(value: min(max(value, 0), 1))
                .tint(value > 0.85 ? Color.wayneRed : Color.wayneAccent)
        }
    }

    private func managerButton(_ title: String, icon: String, action: @escaping () async -> Void) -> some View {
        Button {
            Task { await action() }
        } label: {
            HStack {
                Image(systemName: icon)
                Text(title)
            }
            .font(.caption.monospaced())
            .frame(maxWidth: .infinity)
            .padding()
            .background(Color.wayneSurface)
            .foregroundStyle(Color.wayneAccent)
            .clipShape(RoundedRectangle(cornerRadius: 8))
        }
        .disabled(loading)
    }

    private func run(_ action: () async -> String) async {
        loading = true
        result = await action()
        await refresh()
        loading = false
    }

    private func refresh() async {
        status = await WAYNEService.shared.getSystemStatus()
        processes = await WAYNEService.shared.getProcesses()
    }
}
