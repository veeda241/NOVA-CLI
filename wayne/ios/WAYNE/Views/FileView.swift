import SwiftUI

struct FileView: View {
    @State private var query = ""
    @State private var path = "C:\\"
    @State private var files: [FileItem] = []
    @State private var selected: FileItem?
    @State private var preview = ""
    @State private var loading = false

    var body: some View {
        NavigationStack {
            VStack(spacing: 12) {
                HStack {
                    TextField("Search files", text: $query)
                        .textFieldStyle(.roundedBorder)
                        .onSubmit { Task { await search() } }
                    Button(action: { Task { await search() } }) {
                        Image(systemName: "magnifyingglass")
                    }
                    .buttonStyle(.borderedProminent)
                }
                TextField("Folder path", text: $path)
                    .textFieldStyle(.roundedBorder)
                    .onSubmit { Task { await loadDirectory() } }
                HStack {
                    Button("Downloads") { path = NSHomeDirectory() + "/Downloads"; Task { await loadDirectory() } }
                    Button("Documents") { path = NSHomeDirectory() + "/Documents"; Task { await loadDirectory() } }
                    Button("Refresh") { Task { await loadDirectory() } }
                }
                .font(.caption)

                if loading {
                    ProgressView().tint(.wayneAccent)
                }

                List(files) { file in
                    Button {
                        Task { await openPreview(file) }
                    } label: {
                        HStack(spacing: 12) {
                            Image(systemName: icon(for: file.type))
                                .foregroundStyle(color(for: file.type))
                            VStack(alignment: .leading, spacing: 3) {
                                Text(file.name)
                                    .foregroundStyle(.white)
                                Text(file.path)
                                    .font(.caption2)
                                    .foregroundStyle(Color.wayneMuted)
                                    .lineLimit(1)
                                Text(size(file.size))
                                    .font(.caption2)
                                    .foregroundStyle(Color.wayneAmber)
                            }
                        }
                    }
                    .swipeActions {
                        Button(role: .destructive) {
                            Task {
                                await WAYNEService.shared.deleteFile(path: file.path)
                                await loadDirectory()
                            }
                        } label: { Label("Delete", systemImage: "trash") }
                    }
                    .contextMenu {
                        Button("Open") { Task { await WAYNEService.shared.openFile(path: file.path) } }
                        Button("Copy Path") { UIPasteboard.general.string = file.path }
                    }
                }
                .scrollContentBackground(.hidden)
            }
            .padding()
            .background(Color.wayneBackground.ignoresSafeArea())
            .navigationTitle("Files")
            .sheet(item: $selected) { file in
                NavigationStack {
                    ScrollView {
                        Text(preview)
                            .font(.system(.body, design: .monospaced))
                            .foregroundStyle(.white)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding()
                    }
                    .background(Color.wayneBackground)
                    .navigationTitle(file.name)
                    .toolbar {
                        Button("Open") { Task { await WAYNEService.shared.openFile(path: file.path) } }
                    }
                }
            }
            .task { await loadDirectory() }
        }
    }

    private func loadDirectory() async {
        loading = true
        files = await WAYNEService.shared.getFiles(dir: path)
        loading = false
    }

    private func search() async {
        guard !query.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            await loadDirectory()
            return
        }
        loading = true
        files = await WAYNEService.shared.searchFiles(query: query, directory: nil)
        loading = false
    }

    private func openPreview(_ file: FileItem) async {
        selected = file
        preview = await WAYNEService.shared.readFile(path: file.path)
    }

    private func icon(for type: String?) -> String {
        switch type {
        case "image": return "photo"
        case "video": return "film"
        case "audio": return "waveform"
        case "archive": return "archivebox"
        case "code": return "chevron.left.forwardslash.chevron.right"
        case "spreadsheet": return "tablecells"
        case "document": return "doc.text"
        default: return "doc"
        }
    }

    private func color(for type: String?) -> Color {
        switch type {
        case "image": return .wayneGreen
        case "video", "audio": return .wayneAmber
        case "code": return .wayneAccent
        default: return .wayneMuted
        }
    }

    private func size(_ bytes: Int) -> String {
        let units = ["B", "KB", "MB", "GB", "TB"]
        var value = Double(bytes)
        var index = 0
        while value >= 1024 && index < units.count - 1 {
            value /= 1024
            index += 1
        }
        return String(format: "%.1f %@", value, units[index])
    }
}
