@preconcurrency import AVFoundation
@preconcurrency import CoreVideo
import Combine
import CoreImage
import SwiftUI

/// Session and frame state sit on a serial queue, and only one inference is ever in flight.
final class LiveCamera: NSObject, ObservableObject, AVCaptureVideoDataOutputSampleBufferDelegate, @unchecked Sendable {
    @Published private(set) var running = false
    @Published private(set) var starting = false
    @Published private(set) var message = "Tap Start camera for live predictions."
    @Published private(set) var overlayFrame: FaceOverlayFrame?
    @Published private(set) var photoOverlay: FaceOverlayFrame?
    @Published private(set) var latest: InferenceRecord?
    @Published private(set) var usingFrontCamera = true
    @Published private(set) var permissionDenied = false

    let session = AVCaptureSession()
    // serial queue for session work and frame state
    private let queue = DispatchQueue(label: "edu.edgeface.camera", qos: .userInitiated)
    // separate queue for landmarks so they never wait on a classifier
    private let landmarkQueue = DispatchQueue(label: "edu.edgeface.landmarks", qos: .userInitiated)
    private var landmarkInFlight = false
    private var lastLandmarkTime = -Double.infinity
    private var overlayEnabled = true
    private let context = CIContext(options: [.cacheIntermediates: false])
    private let pipeline = FacePipeline()
    private var output = AVCaptureVideoDataOutput()
    private var active = false
    private var inFlight = false
    private var activeAttempt = 0
    private var displayedPolicy: ComputePolicy = .all // main thread policy copy for the stale check
    private var generation = 0
    private var lastFrameTime = -Double.infinity
    private var policy: ComputePolicy = .all
    private var startAttempt = 0 // lifecycle token, main thread only
    private var position: AVCaptureDevice.Position = .front
    // preview and rotation coordinator are only touched on the main queue
    private weak var preview: AVCaptureVideoPreviewLayer?
    private var rotation: AVCaptureDevice.RotationCoordinator?
    private var previewObservation: NSKeyValueObservation?
    private var captureObservation: NSKeyValueObservation?
    private var notifications: [NSObjectProtocol] = []

    override init() {
        super.init()
        // interruptions and runtime errors both stop the session and leave a message
        notifications.append(NotificationCenter.default.addObserver(
            forName: .AVCaptureSessionWasInterrupted, object: session, queue: .main) { [weak self] _ in
            Task { @MainActor [weak self] in
                self?.stop(message: "Camera interrupted. Tap Start camera to resume.")
            }
        })
        notifications.append(NotificationCenter.default.addObserver(
            forName: .AVCaptureSessionRuntimeError, object: session, queue: .main) { [weak self] _ in
            Task { @MainActor [weak self] in
                self?.stop(message: "Camera stopped unexpectedly. Tap Start camera to retry.")
            }
        })
    }
    // observers have to be removed by hand
    deinit {
        notifications.forEach { NotificationCenter.default.removeObserver($0) }
    }

    // permission first, then configuration on the queue
    @MainActor func start(policy: ComputePolicy) async {
        guard !running && !starting else { return }
        startAttempt += 1
        let attempt = startAttempt
        starting = true
        permissionDenied = false
        let allowed: Bool
        switch AVCaptureDevice.authorizationStatus(for: .video) {
        case .authorized: allowed = true
        case .notDetermined: allowed = await AVCaptureDevice.requestAccess(for: .video)
        default: allowed = false
        }
        guard starting, startAttempt == attempt else { return }
        guard allowed else {
            starting = false
            permissionDenied = true
            message = "Camera access is off. Allow it in Settings to use live mode."
            return
        }
        // stop() can invalidate the attempt while the permission dialog is open
        guard starting, startAttempt == attempt else { return }
        message = "Starting camera…"
        queue.async { [self] in
            do {
                try configure()
                activeAttempt = attempt
                active = true
                generation += 1
                lastFrameTime = -Double.infinity
                session.startRunning()
                let started = session.isRunning
                active = started
                // state is pushed back to the main thread
                DispatchQueue.main.async { [weak self] in
                    guard let self, self.startAttempt == attempt else { return }
                    self.starting = false
                    self.running = started
                    self.message = started ? "Looking for one face…" : "Camera could not start. Try again."
                }
            } catch {
                active = false
                DispatchQueue.main.async { [weak self] in
                    guard let self, self.startAttempt == attempt else { return }
                    self.starting = false
                    self.running = false
                    self.message = error.localizedDescription
                }
            }
        }
        setPolicy(policy)
    }

    // bumping the attempt token makes any start in progress stale
    @MainActor func stop(message text: String = "Camera paused.") {
        startAttempt += 1
        starting = false
        running = false
        latest = nil
        overlayFrame = nil
        message = text
        queue.async { [self] in
            active = false
            generation += 1 // drop results from the old run or policy
            if session.isRunning { session.stopRunning() }
        }
    }

    // only while stopped, the input gets rebuilt on the next start
    @MainActor func flip() {
        guard !running && !starting else { return }
        usingFrontCamera.toggle()
        let next: AVCaptureDevice.Position = usingFrontCamera ? .front : .back
        queue.async { [self] in position = next }
        latest = nil
        overlayFrame = nil
        message = "\(usingFrontCamera ? "Front" : "Rear") camera selected. Tap Start camera."
    }

    // photo path, landmarks and classification run independently
    @MainActor func analyzePhoto(data: Data, policy: ComputePolicy) async throws -> InferenceRecord {
        photoOverlay = nil
        photoOverlay = try? await Task.detached(priority: .userInitiated) {
            try FaceLandmarkDetector.detect(data: data)
        }.value
        return try await pipeline.run(data: data, policy: policy)
    }

    // used by the study galleries, no overlay and configurable expression
    @MainActor func analyzeDevelopmentImage(data: Data, policy: ComputePolicy, includeExpression: Bool = true, isPrecropped: Bool = false) async throws -> InferenceRecord {
        try await pipeline.run(data: data, policy: policy, includeExpression: includeExpression, isPrecropped: isPrecropped)
    }

    @MainActor func verifyStudentConversion() async throws -> String { try await pipeline.verifyStudentConversion() }
    @MainActor func verifyExpressionConversion() async throws -> String { try await pipeline.verifyExpressionConversion() }
    @MainActor func clearPhotoOverlay() { photoOverlay = nil }

    // disabling drops the current overlay right away
    @MainActor func setOverlayEnabled(_ enabled: Bool) {
        if !enabled { overlayFrame = nil }
        queue.async { [self] in overlayEnabled = enabled }
    }

    // a policy change invalidates the current result and the generation
    @MainActor func setPolicy(_ value: ComputePolicy) {
        displayedPolicy = value
        latest = nil
        queue.async { [self] in
            policy = value
            generation += 1
        }
    }

    @MainActor func attach(_ layer: AVCaptureVideoPreviewLayer) {
        preview = layer
        layer.session = session
        layer.videoGravity = .resizeAspectFill
    }

    // rebuilds inputs and outputs from scratch, 640x480 is enough for the pipeline
    private func configure() throws {
        session.beginConfiguration()
        defer { session.commitConfiguration() }
        session.inputs.forEach { session.removeInput($0) }
        session.outputs.forEach { session.removeOutput($0) }
        session.sessionPreset = .vga640x480
        guard let device = AVCaptureDevice.default(.builtInWideAngleCamera, for: .video, position: position) else {
            throw CameraError.unavailable
        }
        let input = try AVCaptureDeviceInput(device: device)
        guard session.canAddInput(input) else { throw CameraError.unavailable }
        session.addInput(input)
        let nextOutput = AVCaptureVideoDataOutput()
        // late frames are dropped instead of queued
        nextOutput.alwaysDiscardsLateVideoFrames = true
        nextOutput.videoSettings = [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA]
        nextOutput.setSampleBufferDelegate(self, queue: queue)
        guard session.canAddOutput(nextOutput) else { throw CameraError.unavailable }
        session.addOutput(nextOutput)
        output = nextOutput
        // mirroring is handled per connection, not automatically
        if let connection = nextOutput.connection(with: .video) {
            connection.automaticallyAdjustsVideoMirroring = false
            if connection.isVideoMirroringSupported { connection.isVideoMirrored = false }
        }
        DispatchQueue.main.async { [weak self] in
            self?.configureRotation(device: device, output: nextOutput)
        }
    }

    // rotation comes from the coordinator so the preview stays upright
    @MainActor private func configureRotation(device: AVCaptureDevice, output: AVCaptureVideoDataOutput) {
        guard let preview else { return }
        let coordinator = AVCaptureDevice.RotationCoordinator(device: device, previewLayer: preview)
        rotation = coordinator
        // preview angle plus mirroring for the front camera
        previewObservation = coordinator.observe(\.videoRotationAngleForHorizonLevelPreview, options: [.initial, .new]) {
            [weak self] coordinator, _ in
            let angle = coordinator.videoRotationAngleForHorizonLevelPreview
            let mirrored = device.position == .front
            DispatchQueue.main.async { [weak self] in
                guard let connection = self?.preview?.connection else { return }
                if connection.isVideoRotationAngleSupported(angle) { connection.videoRotationAngle = angle }
                connection.automaticallyAdjustsVideoMirroring = false
                if connection.isVideoMirroringSupported { connection.isVideoMirrored = mirrored }
            }
        }
        // the capture connection rotates the pixels physically, so Vision gets .up
        captureObservation = coordinator.observe(\.videoRotationAngleForHorizonLevelCapture, options: [.initial, .new]) {
            [weak self, weak output] coordinator, _ in
            let angle = coordinator.videoRotationAngleForHorizonLevelCapture
            self?.queue.async {
                guard let connection = output?.connection(with: .video),
                      connection.isVideoRotationAngleSupported(angle) else { return }
                connection.videoRotationAngle = angle
            }
        }
    }

    func captureOutput(_ output: AVCaptureOutput, didOutput sampleBuffer: CMSampleBuffer,
                       from connection: AVCaptureConnection) {
        let now = ProcessInfo.processInfo.systemUptime
        guard active, let buffer = CMSampleBufferGetImageBuffer(sampleBuffer) else { return }
        // landmarks go first, a slow classifier must not block them
        updateLandmarks(buffer: buffer, output: output, now: now)
        // at most two classifications per second, never a backlog
        guard !inFlight, now - lastFrameTime >= 0.5 else { return }
        let input = CIImage(cvPixelBuffer: buffer)
        guard let image = context.createCGImage(input, from: input.extent) else { return }
        inFlight = true
        lastFrameTime = now
        // token and policy are captured for the stale check
        let token = generation
        let attempt = activeAttempt
        let selectedPolicy = policy
        Task { [self] in
            let outcome: Result<InferenceRecord, Error>
            do { outcome = .success(try await pipeline.run(image: image, policy: selectedPolicy)) }
            catch { outcome = .failure(error) }
            queue.async { [self] in
                inFlight = false
                guard active, token == generation else { return }
                DispatchQueue.main.async { [weak self] in
                    // drop the result when the run, policy or session changed meanwhile
                    guard let self, self.running, self.startAttempt == attempt,
                          self.displayedPolicy == selectedPolicy else { return }
                    switch outcome {
                    case .success(let record):
                        var liveRecord = record
                        liveRecord.source = "live-camera"
                        self.latest = liveRecord
                        self.message = "Live predictions · \(record.policy)"
                    case .failure(let error):
                        self.latest = nil // never leave an old prediction over a missing face
                        self.message = error.localizedDescription
                    }
                }
            }
        }
    }

    // capture keeps the buffer alive, reading only happens on the landmark queue
    private struct LandmarkBuffer: @unchecked Sendable { let value: CVPixelBuffer }

    private func updateLandmarks(buffer: CVPixelBuffer, output: AVCaptureOutput, now: Double) {
        // about ten overlays per second is plenty
        guard overlayEnabled, !landmarkInFlight, now - lastLandmarkTime >= 0.1 else { return }
        landmarkInFlight = true
        lastLandmarkTime = now
        let attempt = activeAttempt
        // pin the conversion down before the orientation can change
        let width = CGFloat(CVPixelBufferGetWidth(buffer)), height = CGFloat(CVPixelBufferGetHeight(buffer))
        // buffer coordinates to layer coordinates through three probe points
        func metadataPoint(_ x: CGFloat, _ y: CGFloat) -> CGPoint {
            // the rect must not be zero sized or it counts as empty
            let rect = output.metadataOutputRectConverted(fromOutputRect: CGRect(x:x-0.5,y:y-0.5,width:1,height:1))
            return CGPoint(x:rect.midX,y:rect.midY)
        }
        let origin = metadataPoint(width*0.25,height*0.25)
        let xAxis = metadataPoint(width*0.75,height*0.25)
        let yAxis = metadataPoint(width*0.25,height*0.75)
        // basis vectors from the probe points, scaled back to full width and height
        let a = (xAxis.x-origin.x)*2, b = (xAxis.y-origin.y)*2
        let c = (yAxis.x-origin.x)*2, d = (yAxis.y-origin.y)*2
        let transform = CGAffineTransform(a:a,b:b,c:c,d:d,
            tx:origin.x-0.25*(a+c), ty:origin.y-0.25*(b+d))
        let retained = LandmarkBuffer(value: buffer)
        landmarkQueue.async { [self] in
            let frame = (try? FaceLandmarkDetector.detect(buffer: retained.value))?.mapped(transform)
            queue.async { [self] in
                landmarkInFlight = false
                guard active, overlayEnabled, attempt == activeAttempt else { return }
                DispatchQueue.main.async { [weak self] in
                    guard let self, self.running, self.startAttempt == attempt else { return }
                    self.overlayFrame = frame
                }
            }
        }
    }
}

private enum CameraError: LocalizedError {
    case unavailable
    var errorDescription: String? { "This camera is unavailable. Try the other camera or use Photos." }
}

// preview view with three overlay layers for box, features and points
private final class CameraPreviewView: UIView {
    override class var layerClass: AnyClass { AVCaptureVideoPreviewLayer.self }
    var previewLayer: AVCaptureVideoPreviewLayer { layer as! AVCaptureVideoPreviewLayer }
    private let faceLayer = CAShapeLayer()
    private let featureLayer = CAShapeLayer()
    private let pointLayer = CAShapeLayer()
    // setting the geometry redraws the layers
    var geometry: FaceOverlayFrame? { didSet { redraw() } }

    override init(frame: CGRect) {
        super.init(frame: frame)
        clipsToBounds = true
        // dashed cyan box for the face, solid mint for the features
        faceLayer.strokeColor = UIColor.cyan.withAlphaComponent(0.65).cgColor
        faceLayer.fillColor = UIColor.clear.cgColor
        faceLayer.lineWidth = 1
        faceLayer.lineDashPattern = [5,4]
        featureLayer.strokeColor = UIColor.systemMint.cgColor
        featureLayer.fillColor = UIColor.clear.cgColor
        featureLayer.lineWidth = 1.4
        pointLayer.fillColor = UIColor.cyan.cgColor
        for overlay in [faceLayer,featureLayer,pointLayer] { layer.addSublayer(overlay) }
    }
    required init?(coder: NSCoder) { fatalError("init(coder:) has not been implemented") }
    // layers follow the view bounds
    override func layoutSubviews() {
        super.layoutSubviews()
        for overlay in [faceLayer,featureLayer,pointLayer] { overlay.frame = bounds }
        redraw()
    }
    // paths are rebuilt from the geometry, implicit animations are off
    private func redraw() {
        CATransaction.begin()
        CATransaction.setDisableActions(true)
        defer { CATransaction.commit() }
        guard let geometry else {
            faceLayer.path = nil; featureLayer.path = nil; pointLayer.path = nil
            return
        }
        // normalised points to layer points through the preview conversion
        let paths = FaceOverlayPaths.make(frame: geometry) { [self] point in
            previewLayer.layerPointConverted(fromCaptureDevicePoint: point)
        }
        faceLayer.path = paths.0
        featureLayer.path = paths.1
        pointLayer.path = paths.2
    }
}

// SwiftUI bridge for the preview view
struct CameraPreview: UIViewRepresentable {
    @ObservedObject var camera: LiveCamera
    let showOverlay: Bool
    func makeUIView(context: Context) -> UIView {
        let view = CameraPreviewView()
        camera.attach(view.previewLayer)
        return view
    }
    func updateUIView(_ uiView: UIView, context: Context) {
        (uiView as? CameraPreviewView)?.geometry = showOverlay ? camera.overlayFrame : nil
    }
}
