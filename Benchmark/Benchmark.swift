import Foundation
import CoreML

enum BenchmarkResources {
 static var root: URL { URL(fileURLWithPath:CommandLine.arguments[1]) }
 static func url(forResource name: String, withExtension ext: String?, subdirectory: String? = nil) -> URL? {
  if ext == "mlmodelc" {
   let path=root.appendingPathComponent("Models/MacCompiled/\(name).mlmodelc")
   return FileManager.default.fileExists(atPath:path.path) ? path : nil
  }
  let path=root.appendingPathComponent("Resources").appendingPathComponent(subdirectory ?? "").appendingPathComponent(name + (ext.map{"."+$0} ?? ""))
  return FileManager.default.fileExists(atPath:path.path) ? path : nil
 }
}
struct Example: Decodable { let filename:String; let finalId:String? }
@main struct Benchmark {
 static func main() async throws {
  let root=BenchmarkResources.root
  let out=URL(fileURLWithPath:CommandLine.arguments[2]);try FileManager.default.createDirectory(at:out,withIntermediateDirectories:true)
  let models=root.appendingPathComponent("Models/MacCompiled");try FileManager.default.createDirectory(at:models,withIntermediateDirectories:true)
  for name in ["StudentAttributes","FastFaceAttributes","OpenCVExpression"] {
   let target=models.appendingPathComponent(name+".mlmodelc")
   if !FileManager.default.fileExists(atPath:target.path) {
    print("Compiling \(name)")
    let compiled=try await MLModel.compileModel(at:root.appendingPathComponent("Models/\(name).mlpackage"))
    try FileManager.default.copyItem(at:compiled,to:target)
   }
  }
  let examples=try JSONDecoder().decode([Example].self,from:Data(contentsOf:root.appendingPathComponent("Resources/InternetCandidates/manifest.json")))
  guard examples.count == 20 else { fatalError("Expected frozen 20-image manifest") }
  let inputs=try examples.map { ($0,try Data(contentsOf:root.appendingPathComponent("Resources/InternetCandidates/\($0.filename)"))) }
  let configs:[(String,String,ComputePolicy)]=[("fastface-cpu","fastface",.cpuOnly),("fastface-all","fastface",.all),("student-all","student",.all)]
  var rows=[[String:Any]]();var failures=[[String:Any]]()
  for (label,model,policy) in configs {
   UserDefaults.standard.set(model,forKey:"attributeModel");UserDefaults.standard.set("opencv",forKey:"expressionModel");UserDefaults.standard.set("conservative",forKey:"ageIntervalPolicy")
   let pipeline=FacePipeline()
   let coldStart=ProcessInfo.processInfo.systemUptime
   do { _=try await pipeline.run(data:inputs[0].1,policy:policy,includeExpression:true) }
   catch { print("INITIAL FAILURE \(label): \(error)");throw error }
   let coldMS=(ProcessInfo.processInfo.systemUptime-coldStart)*1000
   for (_,data) in inputs { _=try await pipeline.run(data:data,policy:policy,includeExpression:true) }
   for repeatIndex in 0..<5 {
    for offset in 0..<inputs.count {
     let index=(offset+repeatIndex*7)%inputs.count;let (example,data)=inputs[index]
     do {
      let record=try await pipeline.run(data:data,policy:policy,includeExpression:true)
      let encoded=try JSONEncoder().encode(record);let object=try JSONSerialization.jsonObject(with:encoded)
      rows.append(["configuration":label,"repeat":repeatIndex+1,"filename":example.filename,"finalId":example.finalId ?? "","inference":object,"cold_load_plus_first_inference_ms":coldMS])
     } catch { failures.append(["configuration":label,"repeat":repeatIndex+1,"filename":example.filename,"error":error.localizedDescription]) }
    }
   }
   print("Completed \(label)")
  }
  let result:[String:Any] = ["platform":ProcessInfo.processInfo.operatingSystemVersionString,"processor_count":ProcessInfo.processInfo.processorCount,"physical_memory_bytes":ProcessInfo.processInfo.physicalMemory,"warmup":"one cold call, then one full 20-image pass per configuration","repeats":5,"scope":"Native Mac Core ML with same FacePipeline code and image files; resource lookup only adapted. Not PyTorch source-runtime latency.","rows":rows,"failures":failures]
  try JSONSerialization.data(withJSONObject:result,options:[.prettyPrinted,.sortedKeys]).write(to:out.appendingPathComponent("mac-latency.json"))
  print("SAVED \(rows.count) successful timed inferences; \(failures.count) failures")
 }
}
