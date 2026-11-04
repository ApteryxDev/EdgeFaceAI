import Foundation
import Vision
import CoreGraphics
import ImageIO
import UIKit
import SwiftUI

// pure display geometry in normalised coordinates, never goes into the JSON export
struct FaceFeatureLine: Sendable {
    let points: [CGPoint]
    let closed: Bool
}
// face box plus the individual feature lines
struct FaceFeatureOutline: Sendable {
    let bounds: CGRect
    let lines: [FaceFeatureLine]
}
// all detected faces of one frame
struct FaceOverlayFrame: Sendable {
    let faces: [FaceFeatureOutline]
    // maps every point through a transform, used for the camera coordinate conversion
    func mapped(_ transform: CGAffineTransform) -> FaceOverlayFrame {
        FaceOverlayFrame(faces: faces.map { face in
            FaceFeatureOutline(bounds: face.bounds.applying(transform), lines: face.lines.map {
                FaceFeatureLine(points: $0.points.map { $0.applying(transform) }, closed: $0.closed)
            })
        })
    }
}

// thin Vision wrapper, only returns geometry
enum FaceLandmarkDetector {
    static func detect(image: CGImage, orientation: CGImagePropertyOrientation = .up) throws -> FaceOverlayFrame {
        let request = VNDetectFaceLandmarksRequest()
        try VNImageRequestHandler(cgImage: image, orientation: orientation).perform([request])
        return geometry(from: request.results ?? [])
    }
    static func detect(buffer: CVPixelBuffer) throws -> FaceOverlayFrame {
        let request = VNDetectFaceLandmarksRequest()
        try VNImageRequestHandler(cvPixelBuffer: buffer, orientation: .up).perform([request])
        return geometry(from: request.results ?? [])
    }
    // reads the EXIF orientation before detecting, so rotated photos are not analysed sideways
    static func detect(data: Data) throws -> FaceOverlayFrame {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil),
              let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else { throw PipelineError.invalidImage }
        let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any]
        let value = (properties?[kCGImagePropertyOrientation] as? NSNumber)?.uint32Value ?? 1
        return try detect(image: image, orientation: CGImagePropertyOrientation(rawValue: value) ?? .up)
    }

    // four faces are enough for the overlay, more just gets cluttered
    private static func geometry(from observations: [VNFaceObservation]) -> FaceOverlayFrame {
        FaceOverlayFrame(faces: observations.prefix(4).map { face in
            let box = face.boundingBox
            var lines: [FaceFeatureLine] = []
            if let landmarks = face.landmarks {
                // region plus whether it is drawn as a closed path
                let regions: [(VNFaceLandmarkRegion2D?, Bool)] = [
                    (landmarks.faceContour, false), (landmarks.leftEye, true), (landmarks.rightEye, true),
                    (landmarks.leftEyebrow, false), (landmarks.rightEyebrow, false),
                    (landmarks.nose, false), (landmarks.noseCrest, false),
                    (landmarks.outerLips, true), (landmarks.innerLips, true),
                    (landmarks.leftPupil, false), (landmarks.rightPupil, false)
                ]
                for (region, closed) in regions {
                    guard let region else { continue }
                    // normalised region points to normalised top-left image points
                    let points = region.normalizedPoints.map { point in
                        CGPoint(x: box.minX + CGFloat(point.x) * box.width,
                                y: 1 - (box.minY + CGFloat(point.y) * box.height))
                    }
                    lines.append(FaceFeatureLine(points: points, closed: closed))
                }
            }
            // box has to be flipped the same way, Vision y goes bottom up
            return FaceFeatureOutline(bounds: CGRect(x: box.minX, y: 1 - box.maxY,
                width: box.width, height: box.height), lines: lines)
        })
    }
}

// the live layer and the photo canvas share exactly the same path building
enum FaceOverlayPaths {
    static func make(frame: FaceOverlayFrame, map: (CGPoint) -> CGPoint) -> (CGPath, CGPath, CGPath) {
        let outline = CGMutablePath(), features = CGMutablePath(), dots = CGMutablePath()
        for face in frame.faces {
            // box as four corners
            let b = face.bounds
            let corners = [CGPoint(x:b.minX,y:b.minY), CGPoint(x:b.maxX,y:b.minY),
                           CGPoint(x:b.maxX,y:b.maxY), CGPoint(x:b.minX,y:b.maxY)].map(map)
            if let first = corners.first {
                outline.move(to: first); corners.dropFirst().forEach { outline.addLine(to: $0) }
                outline.closeSubpath()
            }
            for line in face.lines {
                let points = line.points.map(map)
                guard let first = points.first else { continue }
                features.move(to: first)
                points.dropFirst().forEach { features.addLine(to: $0) }
                if line.closed { features.closeSubpath() }
                // a small dot on every landmark point
                for point in points {
                    dots.addEllipse(in: CGRect(x:point.x-1.8,y:point.y-1.8,width:3.6,height:3.6))
                }
            }
        }
        return (outline, features, dots)
    }
}

// photo view with overlay, scales the normalised points to the canvas size
struct AnnotatedPhoto: View {
    let image: UIImage
    let overlay: FaceOverlayFrame?
    // optional result shown as a small label on the picture
    var stats: InferenceRecord? = nil
    var body: some View {
        Image(uiImage: image).resizable().scaledToFit()
            .overlay {
                if let overlay {
                    GeometryReader { _ in
                        Canvas { context, size in
                            let paths = FaceOverlayPaths.make(frame: overlay) {
                                CGPoint(x:$0.x*size.width, y:$0.y*size.height)
                            }
                            context.stroke(Path(paths.0), with:.color(.cyan.opacity(0.7)), lineWidth:1)
                            context.stroke(Path(paths.1), with:.color(.mint), lineWidth:1.3)
                            context.fill(Path(paths.2), with:.color(.cyan))
                        }
                    }
                }
            }
            .overlay(alignment: .bottomLeading) {
                if let stats { OverlayHUD(record: stats) }
            }
            .accessibilityLabel("Selected photo with detected facial feature outlines")
    }
}
