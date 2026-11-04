import Foundation
import CoreGraphics
import CoreImage
import Vision

/// Five-point similarity to the UniFace 128 template from FastFace.
/// Vision returns eye, nose and lip points, which sit differently from the RetinaFace points.
enum FaceAlignment {
    // fixed target landmarks of the 128 crop
    static let reference128: [CGPoint] = [
        CGPoint(x:46.2946,y:51.6963), CGPoint(x:81.5318,y:51.5014),
        CGPoint(x:64.0252,y:71.7366), CGPoint(x:49.5493,y:92.3655), CGPoint(x:78.7299,y:92.2041)
    ]

    // least squares similarity transform from the detected points to the template
    static func similarity(from source: [CGPoint], to target: [CGPoint]) throws -> CGAffineTransform {
        guard source.count == target.count, source.count >= 2 else { throw AlignmentError.landmarks }
        let count = CGFloat(source.count)
        // centroids of both point sets
        let sm = CGPoint(x:source.map(\.x).reduce(0,+)/count, y:source.map(\.y).reduce(0,+)/count)
        let tm = CGPoint(x:target.map(\.x).reduce(0,+)/count, y:target.map(\.y).reduce(0,+)/count)
        var denominator: CGFloat = 0, real: CGFloat = 0, imaginary: CGFloat = 0
        // accumulate the complex sums for scale and rotation
        for (s,t) in zip(source,target) {
            let sx=s.x-sm.x, sy=s.y-sm.y, tx=t.x-tm.x, ty=t.y-tm.y
            denominator += sx*sx+sy*sy
            real += sx*tx+sy*ty
            imaginary += sx*ty-sy*tx
        }
        // degenerate point sets give no usable scale
        guard denominator > 1e-8 else { throw AlignmentError.landmarks }
        let a=real/denominator, b=imaginary/denominator
        // a/b build the rotation and scale, the translation comes from the centroids
        return CGAffineTransform(a:a,b:b,c:-b,d:a,
            tx:tm.x-a*sm.x+b*sm.y, ty:tm.y-b*sm.x-a*sm.y)
    }

    // aligns and crops one face to a square image
    static func crop(image: CGImage, face: VNFaceObservation, context: CIContext, size: Int = 128) throws -> CGImage {
        guard let landmarks=face.landmarks,
              let leftEye=landmarks.leftEye, let rightEye=landmarks.rightEye,
              let nose=landmarks.nose, let lips=landmarks.outerLips else { throw AlignmentError.landmarks }
        let box=face.boundingBox
        // Vision is normalised and origin bottom left, so convert to pixels first
        func pixels(_ region: VNFaceLandmarkRegion2D) -> [CGPoint] {
            region.normalizedPoints.map {
                CGPoint(x:(box.minX+CGFloat($0.x)*box.width)*CGFloat(image.width),
                        y:(1-box.minY-CGFloat($0.y)*box.height)*CGFloat(image.height))
            }
        }
        // mean point of a landmark region
        func centre(_ points: [CGPoint]) throws -> CGPoint {
            guard !points.isEmpty else { throw AlignmentError.landmarks }
            return CGPoint(x:points.map(\.x).reduce(0,+)/CGFloat(points.count),
                           y:points.map(\.y).reduce(0,+)/CGFloat(points.count))
        }
        // sort the eyes left to right so the point order matches the template
        let eyes = try [centre(pixels(leftEye)),centre(pixels(rightEye))].sorted { $0.x < $1.x }
        let mouth=pixels(lips)
        guard let mouthLeft=mouth.min(by:{$0.x<$1.x}),let mouthRight=mouth.max(by:{$0.x<$1.x}) else { throw AlignmentError.landmarks }
        let points = try [eyes[0],eyes[1],centre(pixels(nose)),mouthLeft,mouthRight]
        // the 112 template is just the 128 one shifted 8px left
        let target = size == 112 ? reference128.map { CGPoint(x: $0.x - 8, y: $0.y) } : reference128
        let topLeft=try similarity(from:points,to:target)
        // points are top-left based, Core Image is bottom-left based
        let height=CGFloat(image.height)
        let ciTransform=CGAffineTransform(a:topLeft.a,b:-topLeft.b,c:-topLeft.c,d:topLeft.d,
            tx:topLeft.tx+topLeft.c*height,
            ty:CGFloat(size)-topLeft.d*height-topLeft.ty)
        let bounds=CGRect(x:0,y:0,width:CGFloat(size),height:CGFloat(size))
        let transformed=CIImage(cgImage:image).samplingLinear().transformed(by:ciTransform)
        // black background fills the corners the rotation leaves empty
        let background=CIImage(color:.black).cropped(to:bounds)
        let padded=transformed.composited(over:background).cropped(to:bounds)
        guard let crop=context.createCGImage(padded,from:bounds) else { throw AlignmentError.landmarks }
        return crop
    }
}

enum AlignmentError: LocalizedError {
    case landmarks
    var errorDescription: String? { "Could not align facial features. Face the camera directly and improve the lighting." }
}
