# Privacy and release audit

Prepared September 23, 2026 for iOS 0.2.0 (4), bundle `dev.mlxpeer.worker.samuelreyes`.

- App privacy draft: **Data Not Collected**. There are no developer-side telemetry, analytics, ads, cloud inference, or user accounts. Transfers are only between the user's paired devices. Apple diagnostics and voluntarily submitted public GitHub issues are described in the policy.
- Tracking: no. No tracking domains or collected-data entries in PrivacyInfo.xcprivacy.
- File metadata: `NSPrivacyAccessedAPICategoryFileTimestamp`, reason `C617.1`. FileManager metadata reads validate and resume model files inside the app container. The Release UI cannot access the debug fixture document picker.
- Disk space: `NSPrivacyAccessedAPICategoryDiskSpace`, reason `E174.1`. CompanionStore checks capacity before creating a model transfer and refuses the transfer when space is insufficient; raw disk capacity is not sent to the Mac.
- System boot time and UserDefaults: no direct use found in the compiled app's imported symbols or the app source. No unnecessary declarations added.
- Encryption declaration: `ITSAppUsesNonExemptEncryption = NO`. Pairing randomness, hashes and data protection use Apple's operating-system APIs. There is no custom encryption implementation or app-level encrypted tunnel.
- Release removes the development fixture/self-test launch UI and pairing-code debug report. Documents file sharing is disabled in Release.
- Icon: original vector artwork, opaque 1024 × 1024 PNG in the Xcode AppIcon asset catalog. Matching macOS ICNS included in packaging.
- Initial target: iPhone. Mac Catalyst, iOS-on-Mac and visionOS compatibility are disabled in project settings. Verify corresponding availability switches in App Store Connect.
- MLX Swift is pinned to 0.31.6. MLX, MLX Swift, MLX C, fmt, nlohmann/json and metal-cpp license notices are bundled and accessible in the app.

Primary references checked:
- https://developer.apple.com/documentation/bundleresources/describing-use-of-required-reason-api
- https://developer.apple.com/documentation/bundleresources/app-privacy-configuration/nsprivacyaccessedapitypes/nsprivacyaccessedapitype
- https://developer.apple.com/help/app-store-connect/manage-app-information/add-an-app-icon
- https://developer.apple.com/help/app-store-connect/reference/app-information/screenshot-specifications
- https://developer.apple.com/news/upcoming-requirements/

This source/build audit is preparation, not an App Store validation or approval result. Apple processing and App Review remain separate steps.
