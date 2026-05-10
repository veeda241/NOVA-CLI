import SwiftUI

struct LearningView: View {
    @State private var stats = LearningStats.empty

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                VStack(alignment: .leading, spacing: 8) {
                    Text(String(format: "%.1f", stats.learningScore))
                        .font(.system(size: 44, weight: .bold, design: .monospaced))
                        .foregroundStyle(Color.wayneAccent)
                    Text("Learning Score")
                        .font(.body.monospaced())
                        .foregroundStyle(Color.wayneAmber)
                    ProgressView(value: stats.learningScore, total: 100)
                        .tint(Color.wayneAccent)
                    Text("Based on \(stats.totalInteractions) interactions")
                        .font(.caption.monospaced())
                        .foregroundStyle(.secondary)
                }
                .padding()
                .background(Color.waynePanel)
                .overlay(RoundedRectangle(cornerRadius: 12).stroke(Color.wayneAccent.opacity(0.3), lineWidth: 0.5))

                Text("Learned Preferences")
                    .font(.headline.monospaced())
                    .foregroundStyle(Color.wayneAccent)
                ForEach(stats.learnedPreferences) { pref in
                    HStack {
                        Text(pref.key).foregroundStyle(Color.wayneAccent.opacity(0.8))
                        Spacer()
                        Text(pref.value).foregroundStyle(Color.wayneAmber)
                        Text("\(Int(pref.confidence * 100))%").foregroundStyle(Color.wayneGreen)
                    }
                    .font(.caption.monospaced())
                    .padding()
                    .background(Color.waynePanel)
                }

                Text("Top Topics")
                    .font(.headline.monospaced())
                    .foregroundStyle(Color.wayneAccent)
                ForEach(stats.topTopics) { topic in
                    HStack {
                        Text(topic.topic)
                        Spacer()
                        Text("\(topic.frequency)")
                        Text("\(Int(topic.avgScore * 100))%")
                    }
                    .font(.caption.monospaced())
                    .foregroundStyle(Color.wayneAccent)
                    .padding()
                    .background(Color.waynePanel)
                }
            }
            .padding()
        }
        .background(Color.wayneBackground.ignoresSafeArea())
        .navigationTitle("Learning")
        .task { stats = await WAYNEService.shared.getLearningStats() }
        .refreshable { stats = await WAYNEService.shared.getLearningStats() }
    }
}
