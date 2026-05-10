import Foundation

struct LearningPreference: Codable, Hashable, Identifiable {
    var id: String { key }
    let key: String
    let value: String
    let confidence: Double
    let sampleCount: Int

    enum CodingKeys: String, CodingKey {
        case key, value, confidence
        case sampleCount = "sample_count"
    }
}

struct LearningTopic: Codable, Hashable, Identifiable {
    var id: String { topic }
    let topic: String
    let frequency: Int
    let avgScore: Double

    enum CodingKeys: String, CodingKey {
        case topic, frequency
        case avgScore = "avg_score"
    }
}

struct LearningStats: Codable {
    let totalInteractions: Int
    let averageReward: Double
    let goldenResponsesStored: Int
    let topTopics: [LearningTopic]
    let learnedPreferences: [LearningPreference]
    let learningScore: Double

    enum CodingKeys: String, CodingKey {
        case totalInteractions = "total_interactions"
        case averageReward = "average_reward"
        case goldenResponsesStored = "golden_responses_stored"
        case topTopics = "top_topics"
        case learnedPreferences = "learned_preferences"
        case learningScore = "learning_score"
    }

    static let empty = LearningStats(totalInteractions: 0, averageReward: 0, goldenResponsesStored: 0, topTopics: [], learnedPreferences: [], learningScore: 0)
}
