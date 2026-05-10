import Foundation

struct FileItem: Identifiable, Codable, Hashable {
    var id: String { path }
    let name: String
    let path: String
    let type: String?
    let size: Int
    let modified: String?

    enum CodingKeys: String, CodingKey {
        case name, path, type, size, modified
        case fileName = "file_name"
        case filePath = "file_path"
        case fileType = "file_type"
        case fileSize = "file_size"
        case modifiedAt = "modified_at"
    }

    init(name: String, path: String, type: String?, size: Int, modified: String?) {
        self.name = name
        self.path = path
        self.type = type
        self.size = size
        self.modified = modified
    }

    init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.name = try container.decodeIfPresent(String.self, forKey: .name)
            ?? container.decodeIfPresent(String.self, forKey: .fileName)
            ?? "Unknown"
        self.path = try container.decodeIfPresent(String.self, forKey: .path)
            ?? container.decodeIfPresent(String.self, forKey: .filePath)
            ?? name
        self.type = try container.decodeIfPresent(String.self, forKey: .type)
            ?? container.decodeIfPresent(String.self, forKey: .fileType)
        self.size = try container.decodeIfPresent(Int.self, forKey: .size)
            ?? container.decodeIfPresent(Int.self, forKey: .fileSize)
            ?? 0
        self.modified = try container.decodeIfPresent(String.self, forKey: .modified)
            ?? container.decodeIfPresent(String.self, forKey: .modifiedAt)
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(name, forKey: .name)
        try container.encode(path, forKey: .path)
        try container.encodeIfPresent(type, forKey: .type)
        try container.encode(size, forKey: .size)
        try container.encodeIfPresent(modified, forKey: .modified)
    }
}
