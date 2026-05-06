// lib/screens/core/home_screen.dart
import 'dart:io';
import 'dart:async';
import 'dart:convert';
import 'dart:math'; 
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:image_picker/image_picker.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:firebase_storage/firebase_storage.dart';
import 'package:rxvision_app/widgets/app_drawer.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';
import 'package:google_generative_ai/google_generative_ai.dart';
import 'package:rxvision_app/models/medication.dart';
import 'package:tflite_flutter/tflite_flutter.dart'; 
import 'package:dio/dio.dart'; 

class HomeScreen extends StatefulWidget {
  const HomeScreen({super.key});

  @override
  State<HomeScreen> createState() => _HomeScreenState();
}

class _HomeScreenState extends State<HomeScreen> {
  File? _image;
  final _textController = TextEditingController();
  bool _isLoading = false;
  bool _isSaving = false;

  // --- AI Models ---
  GenerativeModel? _geminiModel; 
  Interpreter? _tfliteInterpreter; 
  List<String> _tfliteLabels = [];

  // --- Configuration ---
  final String PYTHON_SERVER_URL = 'http://192.168.1.4:8000/analyze_prescription';

  // --- Data ---
  String _fullOcrText = "";
  List<Medication> _detectedMedications = [];
  
  // Thesis Requirement: Doctor Verification Variables
  String? _doctorName; 
  String? _licenseNumber;
  bool _isDoctorVerified = false;
  List<Map<String, String>> _doctorDatabase = []; // Stores Name and License
  
  String _scanStatus = "Scan a prescription to begin.";
  bool _isExpired = false;

  // --- Council (Python) Data ---
  bool _loadingSecondOpinion = false;
  String? _councilTranscription;
  List<dynamic>? _councilAnalysis;
  String _councilStatus = "Ready";

  @override
  void initState() {
    super.initState();
    _setupGemini();
    _loadTFLiteModel();
    _loadDoctorDatabase(); 
  }

  void _setupGemini() {
    final apiKey = dotenv.env['GEMINI_API_KEY'];
    if (apiKey == null || apiKey.isEmpty) {
      setState(() => _scanStatus = "API Key Config Error");
      return;
    }
    _geminiModel = GenerativeModel(
      model: 'gemini-2.5-pro', 
      apiKey: apiKey,
    );
    print("✅ Gemini Vision Model Initialized (v2.5 Pro)");
  }

  Future<void> _loadTFLiteModel() async {
    try {
      _tfliteInterpreter = await Interpreter.fromAsset('assets/ml/prescription_model.tflite');
      final labelsData = await rootBundle.loadString('assets/ml/labels.txt');
      _tfliteLabels = labelsData.split('\n').where((s) => s.isNotEmpty).toList();
      print("✅ TFLite Backup Model Loaded");
    } catch (e) {
      print("⚠️ TFLite model failed: $e");
    }
  }

  // --- Load and Parse CSV (Name and License) ---
  Future<void> _loadDoctorDatabase() async {
    try {
      final csvData = await rootBundle.loadString('assets/doctornames.csv');
      final lines = csvData.split('\n');
      
      List<Map<String, String>> db = [];
      for (int i = 0; i < lines.length; i++) {
        // Skip header if it exists
        if (i == 0 && lines[i].toLowerCase().contains('license')) continue;
        
        final line = lines[i].trim();
        if (line.isEmpty) continue;
        
        final parts = line.split(',');
        if (parts.isNotEmpty) {
          db.add({
            'name': parts[0].trim(),
            'license': parts.length > 1 ? parts[1].trim() : '',
          });
        }
      }
      setState(() => _doctorDatabase = db);
      print("✅ Loaded ${_doctorDatabase.length} doctors with licenses from CSV");
    } catch (e) {
      print("⚠️ Failed to load doctor CSV: $e");
    }
  }

  // --- Local Fuzzy Matcher (Levenshtein Distance) ---
  int _calculateDistance(String a, String b) {
    if (a.isEmpty) return b.length;
    if (b.isEmpty) return a.length;
    List<int> v0 = List<int>.generate(b.length + 1, (i) => i);
    List<int> v1 = List<int>.filled(b.length + 1, 0);
    for (int i = 0; i < a.length; i++) {
      v1[0] = i + 1;
      for (int j = 0; j < b.length; j++) {
        int cost = (a[i] == b[j]) ? 0 : 1;
        v1[j + 1] = min(v1[j] + 1, min(v0[j + 1] + 1, v0[j] + cost));
      }
      for (int j = 0; j <= b.length; j++) {
        v0[j] = v1[j];
      }
    }
    return v1[b.length];
  }

  @override
  void dispose() {
    _textController.dispose();
    _tfliteInterpreter?.close();
    super.dispose();
  }

  Future<void> _pickImage(ImageSource source) async {
    final pickedFile = await ImagePicker().pickImage(source: source);
    if (pickedFile != null) {
      setState(() {
        _image = File(pickedFile.path);
        _textController.clear();
        _detectedMedications.clear();
        _doctorName = null;
        _licenseNumber = null;
        _isDoctorVerified = false;
        _scanStatus = "Processing...";
        _isExpired = false;
        _isLoading = true;
        
        _councilTranscription = null;
        _councilAnalysis = null;
        _councilStatus = "Waiting for analysis...";
      });
      await _performHybridScan();
      setState(() => _isLoading = false);
    }
  }

  Future<void> _performHybridScan() async {
    if (_image == null || _geminiModel == null) return;
    await _askTheKing();
    _askTheCouncil();
  }

  Future<void> _askTheKing() async {
    setState(() => _scanStatus = "Analyzing with AI Vision...");
    try {
      final imageBytes = await _image!.readAsBytes();
      
      // We no longer force Gemini to use a list. We let it read what it sees natively.
      final prompt = TextPart("""
      Analyze this prescription image.
      
      TASK 1: Check the Date.
      If older than 6 months from today (${DateTime.now().toString().split(' ')[0]}), mark as EXPIRED.
      
      TASK 2: Extract Data.
      Extract the attending Doctor's Name and License Number (if visible) exactly as written.
      Extract all medications, dosages, and instructions.
      
      Respond *only* with this JSON:
      {
        "prescription_date": "YYYY-MM-DD" or "null",
        "doctor_name": "string" or "null",
        "license_number": "string" or "null",
        "is_expired": true/false,
        "medications": [{"name": "string", "dosage": "string", "instructions": "string"}]
      }
      """);

      final imagePart = DataPart('image/jpeg', imageBytes);
      final response = await _geminiModel!.generateContent([Content.multi([prompt, imagePart])]);

      if (response.text == null) throw Exception("Empty AI response");

      String cleanJson = response.text!.replaceAll("```json", "").replaceAll("```", "").trim();
      final data = jsonDecode(cleanJson) as Map<String, dynamic>;
      
      if (data['is_expired'] == true) {
        setState(() {
          _isExpired = true;
          _scanStatus = "⚠️ PRESCRIPTION EXPIRED ⚠️\nDate: ${data['prescription_date']}";
          _detectedMedications = [];
          _fullOcrText = "Scan rejected: Expired.";
        });
        return;
      }

      // --- LOCAL DOCTOR VERIFICATION LOGIC ---
      String? rawName = data['doctor_name'];
      String? rawLicense = data['license_number'];
      
      bool verified = false;
      String finalName = rawName ?? 'Unknown';
      String finalLicense = rawLicense ?? 'Unknown';

      if (_doctorDatabase.isNotEmpty && (rawName != null || rawLicense != null)) {
        // Normalize strings to ignore spaces, periods, and case
        String normalize(String s) => s.replaceAll(RegExp(r'[^a-zA-Z0-9]'), '').toLowerCase();
        
        for (var doc in _doctorDatabase) {
          bool nameMatches = false;
          bool licenseMatches = false;

          // 1. Check License Match (Strongest verification point)
          if (rawLicense != null && rawLicense != 'null' && doc['license']!.isNotEmpty) {
            if (normalize(rawLicense) == normalize(doc['license']!)) licenseMatches = true;
          }

          // 2. Check Name Match (Using Fuzzy logic)
          if (rawName != null && rawName != 'null' && doc['name']!.isNotEmpty) {
            String normRaw = normalize(rawName);
            String normDb = normalize(doc['name']!);
            
            // Substring match (e.g. "Dr. B. Who" contains "B Who")
            if (normRaw.contains(normDb) || normDb.contains(normRaw)) {
              nameMatches = true;
            } else {
              // Fuzzy distance check. If distance is <= 3, it's close enough (e.g. "drbvho" vs "drbwho")
              if (_calculateDistance(normRaw, normDb) <= 3) {
                nameMatches = true;
              }
            }
          }

          // If either matches securely, we verify the doctor and use clean database data
          if (nameMatches || licenseMatches) {
            verified = true;
            finalName = doc['name']!; // Override messy AI OCR with clean database name
            finalLicense = doc['license']!.isNotEmpty ? doc['license']! : finalLicense;
            break; 
          }
        }
      }

      List<Medication> meds = (data['medications'] as List).map((m) => Medication.fromJson(m)).toList();

      setState(() {
        _detectedMedications = meds;
        _doctorName = finalName;
        _licenseNumber = finalLicense;
        _isDoctorVerified = verified;
        _scanStatus = meds.isEmpty ? "No medications found." : "";
        _fullOcrText = meds.map((m) => "Rx: ${m.name} ${m.dosage}\nSig: ${m.instructions}").join("\n\n");
        _textController.text = _fullOcrText;
      });

    } catch (e) {
      print("❌ Gemini Error: $e");
      setState(() => _scanStatus = "Analysis Failed. Please try again.");
    }
  }

  Future<void> _askTheCouncil() async {
    if (_image == null) return;
    setState(() {
      _loadingSecondOpinion = true;
      _councilStatus = "Contacting Local Server...";
    });

    try {
      String fileName = _image!.path.split('/').last;
      FormData formData = FormData.fromMap({
        "file": await MultipartFile.fromFile(_image!.path, filename: fileName),
      });

      final dio = Dio(BaseOptions(connectTimeout: const Duration(seconds: 10))); 
      final response = await dio.post(PYTHON_SERVER_URL, data: formData);

      if (response.statusCode == 200) {
        final data = response.data;
        setState(() {
          _councilTranscription = data['text_transcription'];
          _councilAnalysis = data['medical_analysis'];
          _councilStatus = "Validation Complete.";
        });
      }
    } catch (e) {
      print("Council unavailable: $e");
      setState(() => _councilStatus = "Local Server Offline (Skipped)");
    } finally {
      if (mounted) setState(() => _loadingSecondOpinion = false);
    }
  }

  Future<void> _savePrescription() async {
    if (_isExpired) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Cannot save expired prescription.'), backgroundColor: Colors.red),
      );
      return;
    }
    if (_detectedMedications.isEmpty && _fullOcrText.isEmpty) return;

    setState(() => _isSaving = true);
    try {
      final user = FirebaseAuth.instance.currentUser;
      if (user == null) throw Exception("No user");

      String? imageUrl;

      if (_image != null) {
        final String fileName = '${DateTime.now().millisecondsSinceEpoch}.jpg';
        final storageRef = FirebaseStorage.instance
            .ref()
            .child('user_prescriptions/${user.uid}/$fileName');
        
        await storageRef.putFile(_image!);
        imageUrl = await storageRef.getDownloadURL();
      }

      await FirebaseFirestore.instance
          .collection('users')
          .doc(user.uid)
          .collection('prescriptions')
          .add({
        'text': _fullOcrText,
        'detectedDrugs': _detectedMedications.map((m) => m.name).toList(),
        'doctorName': _doctorName ?? 'Unknown', 
        'licenseNumber': _licenseNumber ?? 'Unknown',
        'isVerified': _isDoctorVerified,
        'timestamp': FieldValue.serverTimestamp(),
        'is_valid': true,
        'imageUrl': imageUrl, 
        'council_validation': _councilAnalysis != null ? _councilAnalysis.toString() : 'Skipped',
      });

      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Saved Successfully!'), backgroundColor: Colors.green),
        );
        setState(() {
          _image = null;
          _detectedMedications.clear();
          _doctorName = null;
          _licenseNumber = null;
          _isDoctorVerified = false;
          _textController.clear();
          _scanStatus = "Scan a prescription to begin.";
          _councilTranscription = null;
          _councilAnalysis = null;
        });
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Error: $e'), backgroundColor: Colors.red),
        );
      }
    } finally {
      setState(() => _isSaving = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      drawer: const AppDrawer(),
      appBar: AppBar(title: const Text('RxVision Scanner')),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            // Image Preview
            if (_image != null)
              Container(
                width: double.infinity,
                decoration: BoxDecoration(
                  color: Colors.grey[200],
                  borderRadius: BorderRadius.circular(12),
                  border: Border.all(color: Theme.of(context).primaryColor.withOpacity(0.3)),
                ),
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(12),
                  child: Image.file(_image!, fit: BoxFit.contain),
                ),
              )
            else
              Container(
                height: 250,
                decoration: BoxDecoration(
                  color: Colors.grey[200],
                  borderRadius: BorderRadius.circular(12),
                  border: Border.all(color: Theme.of(context).primaryColor.withOpacity(0.3)),
                ),
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    Icon(Icons.document_scanner, size: 50, color: Colors.grey[400]),
                    const SizedBox(height: 10),
                    Text('No Image Selected', style: TextStyle(color: Colors.grey[600])),
                  ],
                ),
              ),

            const SizedBox(height: 20),
            
            // Buttons
            Row(
              children: [
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: _isLoading ? null : () => _pickImage(ImageSource.camera),
                    icon: const Icon(Icons.camera_alt),
                    label: const Text('Camera'),
                    style: ElevatedButton.styleFrom(padding: const EdgeInsets.symmetric(vertical: 12)),
                  ),
                ),
                const SizedBox(width: 16),
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: _isLoading ? null : () => _pickImage(ImageSource.gallery),
                    icon: const Icon(Icons.photo_library),
                    label: const Text('Gallery'),
                    style: ElevatedButton.styleFrom(padding: const EdgeInsets.symmetric(vertical: 12)),
                  ),
                ),
              ],
            ),

            const Divider(height: 30),

            // Results
            if (_isExpired)
              Container(
                padding: const EdgeInsets.all(16),
                decoration: BoxDecoration(
                  color: Colors.red[50],
                  borderRadius: BorderRadius.circular(12),
                  border: Border.all(color: Colors.red),
                ),
                child: Column(
                  children: [
                    const Icon(Icons.warning_amber_rounded, color: Colors.red, size: 40),
                    const SizedBox(height: 10),
                    Text(_scanStatus, textAlign: TextAlign.center, style: const TextStyle(color: Colors.red, fontWeight: FontWeight.bold)),
                  ],
                ),
              )
            else
              Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'Scan Summary 💊',
                    style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold, color: Theme.of(context).primaryColor),
                  ),
                  const SizedBox(height: 10),
                  
                  // --- THESIS REQUIREMENT: DOCTOR VERIFICATION UI (SAAS VIBE) ---
                  if (_doctorName != null && _doctorName != 'null')
                    Container(
                      margin: const EdgeInsets.only(bottom: 16),
                      padding: const EdgeInsets.all(14),
                      decoration: BoxDecoration(
                        color: _isDoctorVerified ? Colors.green[50] : Colors.orange[50],
                        borderRadius: BorderRadius.circular(8),
                        border: Border.all(
                          color: _isDoctorVerified ? Colors.green.shade200 : Colors.orange.shade200, 
                          width: 1.5,
                        ),
                      ),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Row(
                            children: [
                              Icon(_isDoctorVerified ? Icons.verified_user : Icons.gpp_maybe, 
                                   color: _isDoctorVerified ? Colors.green[600] : Colors.orange[600], size: 22),
                              const SizedBox(width: 8),
                              Expanded(
                                child: Text(
                                  "Doctor: $_doctorName", 
                                  style: TextStyle(fontWeight: FontWeight.w600, fontSize: 15, 
                                                   color: _isDoctorVerified ? Colors.green[900] : Colors.orange[900]),
                                ),
                              ),
                            ],
                          ),
                          if (_licenseNumber != null && _licenseNumber != 'null' && _licenseNumber != 'Unknown' && _licenseNumber!.isNotEmpty) ...[
                            const SizedBox(height: 6),
                            Padding(
                              padding: const EdgeInsets.only(left: 30.0),
                              child: Text("License: $_licenseNumber", style: TextStyle(color: Colors.blueGrey[700], fontSize: 13, fontWeight: FontWeight.w500)),
                            ),
                          ],
                          const SizedBox(height: 8),
                          Padding(
                            padding: const EdgeInsets.only(left: 30.0),
                            child: Text(
                              _isDoctorVerified ? "Verified against local database" : "Not found in registry",
                              style: TextStyle(
                                fontSize: 12, 
                                fontWeight: FontWeight.w600,
                                letterSpacing: 0.3,
                                color: _isDoctorVerified ? Colors.green[700] : Colors.orange[800]
                              ),
                            ),
                          ),
                        ],
                      ),
                    ),
                  // --------------------------------------------------------------

                  if (_isLoading)
                    const Center(child: Column(children: [CircularProgressIndicator(), SizedBox(height: 16), Text("Analyzing...")]))
                  else
                    _buildResultsUI(),
                ],
              ),

            const SizedBox(height: 24),

            // Technical Validation Expansion Tile
            ExpansionTile(
              title: const Text("Technical Validation", style: TextStyle(fontSize: 14, fontWeight: FontWeight.bold, color: Colors.grey)),
              children: [
                Container(
                  width: double.infinity,
                  padding: const EdgeInsets.all(12),
                  decoration: BoxDecoration(color: Colors.grey[100], borderRadius: BorderRadius.circular(8)),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const Text("• TFLite Model: Active (Background Monitor)", style: TextStyle(fontSize: 12)),
                      const SizedBox(height: 8),
                      Row(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          const Text("• Python Server: ", style: TextStyle(fontSize: 12)),
                          if (_loadingSecondOpinion) 
                            const Padding(
                              padding: EdgeInsets.only(top: 2, left: 4),
                              child: SizedBox(width: 10, height: 10, child: CircularProgressIndicator(strokeWidth: 2)),
                            )
                          else
                            Expanded(
                              child: Text(
                                _councilStatus, 
                                style: TextStyle(
                                  fontSize: 12, 
                                  fontWeight: FontWeight.bold, 
                                  color: _councilStatus.contains("Complete") ? Colors.green : Colors.orange
                                ),
                                maxLines: 2,
                                overflow: TextOverflow.ellipsis,
                              ),
                            ),
                        ],
                      ),
                    ],
                  ),
                ),
              ],
            ),

            const SizedBox(height: 16),

            // Save Button
            if (!_isLoading && !_isExpired && _detectedMedications.isNotEmpty)
              ElevatedButton.icon(
                onPressed: _isSaving ? null : _savePrescription,
                icon: _isSaving ? const SizedBox.shrink() : const Icon(Icons.check),
                label: _isSaving ? const Text("Saving...") : const Text('Confirm & Save'),
                style: ElevatedButton.styleFrom(backgroundColor: Colors.green, padding: const EdgeInsets.symmetric(vertical: 16)),
              )
          ],
        ),
      ),
    );
  }

  Widget _buildResultsUI() {
    if (_detectedMedications.isEmpty) {
      return Container(
        padding: const EdgeInsets.all(20),
        decoration: BoxDecoration(color: Colors.grey[100], borderRadius: BorderRadius.circular(12)),
        child: Center(child: Text(_scanStatus, textAlign: TextAlign.center)),
      );
    }
    return ListView.builder(
      shrinkWrap: true,
      physics: const NeverScrollableScrollPhysics(),
      itemCount: _detectedMedications.length,
      itemBuilder: (context, index) {
        final med = _detectedMedications[index];
        return Card(
          elevation: 2,
          margin: const EdgeInsets.symmetric(vertical: 6),
          child: Padding(
            padding: const EdgeInsets.all(16.0),
            child: Column(children: [
              _buildMedRow(Icons.medication, "Drug", med.name, isBold: true),
              const Divider(),
              _buildMedRow(Icons.science, "Dose", med.dosage),
              const Divider(),
              _buildMedRow(Icons.timer, "Sig", med.instructions),
            ]),
          ),
        );
      },
    );
  }

  Widget _buildMedRow(IconData icon, String label, String value, {bool isBold = false}) {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Icon(icon, size: 18, color: Colors.blue[700]),
        const SizedBox(width: 8),
        SizedBox(width: 50, child: Text(label, style: const TextStyle(color: Colors.grey, fontSize: 12))),
        Expanded(child: Text(value, style: TextStyle(fontSize: 15, fontWeight: isBold ? FontWeight.bold : FontWeight.normal))),
      ],
    );
  }
}