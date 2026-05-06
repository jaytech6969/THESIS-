// lib/main.dart
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:firebase_core/firebase_core.dart';
import 'package:rxvision_app/app.dart';
import 'firebase_options.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  
  // Load environment variables (for Gemini API key)
  await dotenv.load(fileName: ".env");

  try {
    // Check if Firebase is already initialized
    if (Firebase.apps.isEmpty) {
      print("Attempting to initialize Firebase...");
      await Firebase.initializeApp(
        options: DefaultFirebaseOptions.currentPlatform,
      );
      print("✅ Firebase Initialized Successfully.");
    } else {
      print("ℹ️ Firebase was already initialized. Skipping.");
    }
  } catch (e) {
    // Specifically catch the Duplicate App error and ignore it
    if (e.toString().contains("duplicate-app")) {
      print("ℹ️ Firebase Initialization: App already exists (Safe to ignore).");
    } else {
      print("❌ ERROR DURING FIREBASE INITIALIZATION: $e");
    }
  }
  
  runApp(const RxVisionApp());
}