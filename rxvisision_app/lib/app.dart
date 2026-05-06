// lib/app.dart
import 'package:flutter/material.dart';
import 'package:rxvision_app/config/theme.dart';
import 'package:rxvision_app/screens/auth/auth_gate.dart';

class RxVisionApp extends StatelessWidget {
  const RxVisionApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'RxVision',
      theme: AppTheme.lightTheme,
      debugShowCheckedModeBanner: false,
      home: const AuthGate(),
    );
  }
}