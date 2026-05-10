import SwiftUI

struct TaskView: View {
    @State private var tasks: [TaskItem] = []
    @State private var title = ""

    var body: some View {
        VStack {
            HStack {
                TextField("New task", text: $title)
                    .font(.body.monospaced())
                    .padding(10)
                    .background(Color.waynePanel)
                Button("Add") {
                    Task {
                        await WAYNEService.shared.createTask(title: title, priority: "medium")
                        title = ""
                        await refresh()
                    }
                }
                .foregroundStyle(Color.wayneAccent)
            }
            .padding()
            List {
                ForEach(tasks) { task in
                    HStack {
                        Button {
                            Task {
                                await WAYNEService.shared.toggleTask(id: task.id)
                                await refresh()
                            }
                        } label: {
                            Image(systemName: task.completed ? "checkmark.square.fill" : "square")
                        }
                        Text(task.title)
                            .strikethrough(task.completed)
                            .font(.body.monospaced())
                        Spacer()
                        Text(task.priority)
                            .font(.caption.monospaced())
                            .foregroundStyle(task.priority == "high" ? Color.wayneRed : task.priority == "medium" ? Color.wayneAmber : Color.wayneGreen)
                    }
                    .listRowBackground(Color.wayneSurface)
                }
                .onDelete { offsets in
                    Task {
                        for index in offsets { await WAYNEService.shared.deleteTask(id: tasks[index].id) }
                        await refresh()
                    }
                }
            }
            .scrollContentBackground(.hidden)
        }
        .background(Color.wayneBackground.ignoresSafeArea())
        .navigationTitle("Tasks")
        .task { await refresh() }
        .refreshable { await refresh() }
    }

    private func refresh() async {
        tasks = await WAYNEService.shared.getTasks()
    }
}
