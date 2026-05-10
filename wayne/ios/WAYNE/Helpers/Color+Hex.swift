import SwiftUI

extension Color {
    init(hex: String) {
        let scanner = Scanner(string: hex.trimmingCharacters(in: CharacterSet.alphanumerics.inverted))
        var value: UInt64 = 0
        scanner.scanHexInt64(&value)
        let r = Double((value >> 16) & 0xff) / 255
        let g = Double((value >> 8) & 0xff) / 255
        let b = Double(value & 0xff) / 255
        self.init(red: r, green: g, blue: b)
    }

    static let wayneBackground = Color(hex: "060c14")
    static let wayneSurface = Color(hex: "0a1628")
    static let waynePanel = Color(hex: "0d1f38")
    static let wayneAccent = Color(hex: "00d4ff")
    static let wayneAmber = Color(hex: "ffaa00")
    static let wayneGreen = Color(hex: "00ff88")
    static let wayneRed = Color(hex: "ff4455")
    static let wayneMuted = Color(hex: "4a7a9b")
}
