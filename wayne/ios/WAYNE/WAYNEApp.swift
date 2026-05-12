import SwiftUI
import UserNotifications

@main
struct WAYNEApp: App {
    @UIApplicationDelegateAdaptor(AppDelegate.self) var appDelegate
    @StateObject private var wakeWordService = WakeWordService()
    @StateObject private var connectionService = ConnectionService()
    @State private var showBootScreen = false

    var body: some Scene {
        WindowGroup {
            ZStack {
                ContentView(connection: connectionService)
                    .preferredColorScheme(.dark)
                if showBootScreen {
                    BootScreenView(isShowing: $showBootScreen)
                        .transition(.opacity)
                }
            }
            .onAppear {
                wakeWordService.startPassiveListening()
            }
            .onChange(of: wakeWordService.wakeDetected) { detected in
                if detected {
                    withAnimation { showBootScreen = true }
                    wakeWordService.wakeDetected = false
                }
            }
        }
    }
}

final class AppDelegate: NSObject, UIApplicationDelegate, UNUserNotificationCenterDelegate {
    func application(_ application: UIApplication, didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]? = nil) -> Bool {
        UNUserNotificationCenter.current().delegate = self
        UNUserNotificationCenter.current().requestAuthorization(options: [.alert, .sound, .badge]) { granted, _ in
            if granted {
                DispatchQueue.main.async { application.registerForRemoteNotifications() }
            }
        }
        Task { await WAYNEService.shared.registerPhone(pushToken: nil) }
        return true
    }

    func application(_ application: UIApplication, didRegisterForRemoteNotificationsWithDeviceToken deviceToken: Data) {
        let token = deviceToken.map { String(format: "%02x", $0) }.joined()
        Task { await WAYNEService.shared.registerPhone(pushToken: token) }
    }

    func userNotificationCenter(_ center: UNUserNotificationCenter, willPresent notification: UNNotification) async -> UNNotificationPresentationOptions {
        [.banner, .sound, .list]
    }
}
