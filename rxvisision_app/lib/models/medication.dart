// lib/models/medication.dart
class Medication {
  final String name;
  final String dosage;
  final String instructions;

  Medication({
    required this.name,
    required this.dosage,
    required this.instructions,
  });

  // Factory constructor to create a Medication from JSON
  factory Medication.fromJson(Map<String, dynamic> json) {
    return Medication(
      name: json['name'] ?? 'N/A',
      dosage: json['dosage'] ?? 'N/A',
      instructions: json['instructions'] ?? 'N/A',
    );
  }
}