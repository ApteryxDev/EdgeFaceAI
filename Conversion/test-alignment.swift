import Foundation
import CoreGraphics

@main struct AlignmentChecks {
 static func main() throws {
  let reference=FaceAlignment.reference128
  let identity=try FaceAlignment.similarity(from:reference,to:reference)
  for p in reference { let q=p.applying(identity); precondition(hypot(q.x-p.x,q.y-p.y)<1e-8) }
  let known=CGAffineTransform(a:1.5,b:0.5,c:-0.5,d:1.5,tx:35,ty:-12)
  let source=reference.map{$0.applying(known)}
  let recovered=try FaceAlignment.similarity(from:source,to:reference)
  for (p,target) in zip(source,reference) {let q=p.applying(recovered);precondition(hypot(q.x-target.x,q.y-target.y)<1e-7)}
  do { _=try FaceAlignment.similarity(from:[.zero,.zero],to:[.zero,.zero]);fatalError("Degenerate points accepted") }
  catch AlignmentError.landmarks {}
  print("PASS: identity, rotated/scaled/translated five-point recovery, degenerate-point rejection")
 }
}
