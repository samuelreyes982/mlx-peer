// Original MLX Peer vector artwork. Rebuild with: swift scripts/generate_icons.swift
import AppKit
import Foundation
import ImageIO
import UniformTypeIdentifiers

let root = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
let ios = root.appendingPathComponent("ios/App/Assets.xcassets/AppIcon.appiconset")
let mark = root.appendingPathComponent("ios/App/Assets.xcassets/PeerMark.imageset")
let mac = root.appendingPathComponent("macos/AppIcon.iconset")
for directory in [ios, mark, mac] { try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true) }

func color(_ rgb: UInt32, _ alpha: CGFloat = 1) -> NSColor {
    NSColor(srgbRed: CGFloat((rgb >> 16) & 255) / 255, green: CGFloat((rgb >> 8) & 255) / 255, blue: CGFloat(rgb & 255) / 255, alpha: alpha)
}
func rounded(_ rect: NSRect, _ radius: CGFloat, _ fill: NSColor) {
    fill.setFill(); NSBezierPath(roundedRect: rect, xRadius: radius, yRadius: radius).fill()
}
func stroke(_ points: [NSPoint], width: CGFloat, color: NSColor) {
    let p = NSBezierPath(); p.move(to: points[0]); for point in points.dropFirst() { p.line(to: point) }
    p.lineWidth = width; p.lineCapStyle = .round; p.lineJoinStyle = .round; color.setStroke(); p.stroke()
}
func render(_ pixels: Int, macStyle: Bool, to destination: URL) throws {
    let context = CGContext(data: nil, width: pixels, height: pixels, bitsPerComponent: 8, bytesPerRow: 0, space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: (macStyle ? CGImageAlphaInfo.premultipliedLast : .noneSkipLast).rawValue)!
    NSGraphicsContext.saveGraphicsState()
    NSGraphicsContext.current = NSGraphicsContext(cgContext: context, flipped: false)
    let transform = AffineTransform(scale: CGFloat(pixels) / 1024); (transform as NSAffineTransform).concat()
    if macStyle {
        let inset = AffineTransform(translationByX: 60, byY: 60); (inset as NSAffineTransform).concat()
        (AffineTransform(scale: 904 / 1024) as NSAffineTransform).concat()
    }
    let background = NSBezierPath(roundedRect: NSRect(x: 0, y: 0, width: 1024, height: 1024), xRadius: macStyle ? 220 : 0, yRadius: macStyle ? 220 : 0)
    NSGradient(starting: color(0x243638), ending: color(0x10161D))!.draw(in: background, angle: -65)
    rounded(NSRect(x: 161, y: 374, width: 493, height: 337), 60, color(0xB4F474))
    rounded(NSRect(x: 188, y: 401, width: 439, height: 283), 34, color(0x16272C))
    // A small graph stands for the shared model, never a cloud service.
    stroke([NSPoint(x: 299, y: 477), NSPoint(x: 407, y: 603), NSPoint(x: 515, y: 477)], width: 19, color: color(0xB4F474, 0.68))
    for p in [NSPoint(x: 299, y: 477), NSPoint(x: 407, y: 603), NSPoint(x: 515, y: 477)] {
        rounded(NSRect(x: p.x - 24, y: p.y - 24, width: 48, height: 48), 24, color(0xD9FFAD))
    }
    rounded(NSRect(x: 124, y: 332, width: 567, height: 26), 13, color(0xB4F474))
    // The continuous cable joins the two devices.
    let cable = NSBezierPath(); cable.move(to: NSPoint(x: 404, y: 332))
    cable.line(to: NSPoint(x: 404, y: 289))
    cable.curve(to: NSPoint(x: 478, y: 215), controlPoint1: NSPoint(x: 404, y: 248), controlPoint2: NSPoint(x: 437, y: 215))
    cable.line(to: NSPoint(x: 652, y: 215))
    cable.curve(to: NSPoint(x: 726, y: 289), controlPoint1: NSPoint(x: 693, y: 215), controlPoint2: NSPoint(x: 726, y: 248))
    cable.line(to: NSPoint(x: 726, y: 332)); cable.lineWidth = 26; cable.lineCapStyle = .round
    color(0xEAF6EA).setStroke(); cable.stroke()
    rounded(NSRect(x: 606, y: 320, width: 240, height: 436), 59, color(0x10191F))
    rounded(NSRect(x: 618, y: 332, width: 216, height: 412), 49, color(0xF0F8F0))
    rounded(NSRect(x: 641, y: 355, width: 170, height: 366), 29, color(0x213439))
    rounded(NSRect(x: 695, y: 688, width: 62, height: 13), 6.5, color(0xF0F8F0))
    rounded(NSRect(x: 697, y: 502, width: 58, height: 58), 29, color(0xB4F474))
    rounded(NSRect(x: 699, y: 374, width: 54, height: 8), 4, color(0xF0F8F0, 0.8))
    NSGraphicsContext.restoreGraphicsState()
    let output = CGImageDestinationCreateWithURL(destination as CFURL, UTType.png.identifier as CFString, 1, nil)!
    CGImageDestinationAddImage(output, context.makeImage()!, nil)
    guard CGImageDestinationFinalize(output) else { fatalError("Could not write icon") }
}
try render(1024, macStyle: false, to: ios.appendingPathComponent("AppIcon.png"))
try render(256, macStyle: false, to: mark.appendingPathComponent("PeerMark.png"))
for size in [16, 32, 128, 256, 512] {
    try render(size, macStyle: true, to: mac.appendingPathComponent("icon_\(size)x\(size).png"))
    try render(size * 2, macStyle: true, to: mac.appendingPathComponent("icon_\(size)x\(size)@2x.png"))
}
let info: [String: Any] = ["author": "xcode", "version": 1]
for (directory, images) in [
    (ios, [["filename": "AppIcon.png", "idiom": "universal", "platform": "ios", "size": "1024x1024"]]),
    (mark, [["filename": "PeerMark.png", "idiom": "universal"]])
] {
    try JSONSerialization.data(withJSONObject: ["images": images, "info": info], options: [.prettyPrinted, .sortedKeys]).write(to: directory.appendingPathComponent("Contents.json"))
}
try JSONSerialization.data(withJSONObject: ["info": info], options: [.prettyPrinted]).write(to: ios.deletingLastPathComponent().appendingPathComponent("Contents.json"))
print("Created original iOS and macOS icon assets.")
