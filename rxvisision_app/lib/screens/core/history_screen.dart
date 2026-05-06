// lib/screens/core/history_screen.dart
import 'package:flutter/material.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:intl/intl.dart';

class HistoryScreen extends StatelessWidget {
  const HistoryScreen({super.key});

  Future<void> _deletePrescription(BuildContext context, String docId) async {
    // 1. Show SaaS Styled Confirmation Modal
    final confirm = await showDialog<bool>(
      context: context,
      builder: (ctx) => Dialog(
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
        elevation: 0,
        backgroundColor: Colors.transparent,
        child: Container(
          padding: const EdgeInsets.all(24),
          decoration: BoxDecoration(
            color: Colors.white,
            borderRadius: BorderRadius.circular(16),
            boxShadow: [
              BoxShadow(
                color: Colors.black.withOpacity(0.1),
                blurRadius: 20,
                offset: const Offset(0, 10),
              ),
            ],
          ),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: Colors.red[50],
                  shape: BoxShape.circle,
                ),
                child: const Icon(Icons.delete_outline, color: Colors.red, size: 32),
              ),
              const SizedBox(height: 20),
              const Text(
                "Delete Prescription?",
                style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold, color: Color(0xFF1E293B)),
              ),
              const SizedBox(height: 10),
              const Text(
                "Are you sure you want to delete this prescription? This action cannot be undone.",
                textAlign: TextAlign.center,
                style: TextStyle(fontSize: 14, color: Color(0xFF64748B)),
              ),
              const SizedBox(height: 24),
              Row(
                children: [
                  Expanded(
                    child: TextButton(
                      onPressed: () => Navigator.pop(ctx, false),
                      style: TextButton.styleFrom(
                        padding: const EdgeInsets.symmetric(vertical: 14),
                        backgroundColor: Colors.grey[100],
                        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
                      ),
                      child: const Text("Cancel", style: TextStyle(color: Colors.black54, fontWeight: FontWeight.w600)),
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: ElevatedButton(
                      onPressed: () => Navigator.pop(ctx, true),
                      style: ElevatedButton.styleFrom(
                        padding: const EdgeInsets.symmetric(vertical: 14),
                        backgroundColor: Colors.red,
                        foregroundColor: Colors.white,
                        elevation: 0,
                        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
                      ),
                      child: const Text("Delete", style: TextStyle(fontWeight: FontWeight.w600)),
                    ),
                  ),
                ],
              )
            ],
          ),
        ),
      ),
    );

    if (confirm != true) return;

    // 2. Perform Delete
    final user = FirebaseAuth.instance.currentUser;
    if (user != null) {
      try {
        await FirebaseFirestore.instance
            .collection('users')
            .doc(user.uid)
            .collection('prescriptions')
            .doc(docId)
            .delete();
        
        if (context.mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(content: Text('Record deleted successfully'), backgroundColor: Colors.grey),
          );
        }
      } catch (e) {
        if (context.mounted) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(content: Text('Error deleting: $e'), backgroundColor: Colors.red),
          );
        }
      }
    }
  }

  void _viewPrescriptionDetails(BuildContext context, Map<String, dynamic> data) {
    final text = data['text'] as String? ?? 'No text available';
    final imageUrl = data['imageUrl'] as String?;
    final detectedDrugs = (data['detectedDrugs'] as List<dynamic>?)?.join(', ') ?? 'Unknown';
    
    // --- Extract Doctor's Data ---
    final doctorName = data['doctorName'] as String? ?? 'Unknown';
    final licenseNumber = data['licenseNumber'] as String? ?? 'Unknown';
    final isVerified = data['isVerified'] as bool? ?? false;

    showDialog(
      context: context,
      builder: (context) => Dialog(
        insetPadding: const EdgeInsets.all(12), // Maximize space
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
        child: Container(
          width: double.infinity,
          height: MediaQuery.of(context).size.height * 0.85,
          decoration: BoxDecoration(
            color: Colors.white,
            borderRadius: BorderRadius.circular(16),
          ),
          child: Column(
            children: [
              // Header
              Padding(
                padding: const EdgeInsets.all(16.0),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    const Text('Prescription Details', style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold)),
                    IconButton(
                      icon: const Icon(Icons.close),
                      onPressed: () => Navigator.pop(context),
                    ),
                  ],
                ),
              ),
              const Divider(height: 1),
              
              // Body
              Expanded(
                child: SingleChildScrollView(
                  padding: const EdgeInsets.all(16),
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      if (imageUrl != null) ...[
                        const Text("Original Image", style: TextStyle(fontWeight: FontWeight.bold, color: Colors.grey)),
                        const SizedBox(height: 8),
                        Container(
                          height: 400, // Fixed height container for the image area
                          width: double.infinity,
                          decoration: BoxDecoration(
                            color: Colors.black, // Dark background for better contrast
                            borderRadius: BorderRadius.circular(8),
                            border: Border.all(color: Colors.grey.shade300),
                          ),
                          child: ClipRRect(
                            borderRadius: BorderRadius.circular(8),
                            child: InteractiveViewer(
                              minScale: 0.5,
                              maxScale: 4.0,
                              child: Image.network(
                                imageUrl,
                                fit: BoxFit.contain, // ADJUSTS TO ACTUAL SIZE WITHIN BOUNDS
                                loadingBuilder: (ctx, child, progress) {
                                  if (progress == null) return child;
                                  return const Center(child: CircularProgressIndicator(color: Colors.white));
                                },
                                errorBuilder: (ctx, err, stack) => const Center(
                                  child: Column(
                                    mainAxisAlignment: MainAxisAlignment.center,
                                    children: [
                                      Icon(Icons.broken_image, color: Colors.white54, size: 40),
                                      Text("Image not found", style: TextStyle(color: Colors.white54))
                                    ],
                                  ),
                                ),
                              ),
                            ),
                          ),
                        ),
                        const SizedBox(height: 20),
                      ],

                      // --- DISPLAY DOCTOR NAME AND VERIFICATION (SAAS VIBE) ---
                      const Text("Attending Doctor", style: TextStyle(fontWeight: FontWeight.bold, color: Colors.blue)),
                      const SizedBox(height: 8),
                      if (doctorName != 'Unknown' && doctorName != 'null')
                        Container(
                          margin: const EdgeInsets.only(bottom: 16),
                          padding: const EdgeInsets.all(14),
                          decoration: BoxDecoration(
                            color: isVerified ? Colors.green[50] : Colors.orange[50],
                            borderRadius: BorderRadius.circular(8),
                            border: Border.all(
                              color: isVerified ? Colors.green.shade200 : Colors.orange.shade200, 
                              width: 1.5,
                            ),
                          ),
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Row(
                                children: [
                                  Icon(isVerified ? Icons.verified_user : Icons.gpp_maybe, 
                                       color: isVerified ? Colors.green[600] : Colors.orange[600], size: 22),
                                  const SizedBox(width: 8),
                                  Expanded(
                                    child: Text(
                                      "Doctor: $doctorName", 
                                      style: TextStyle(fontWeight: FontWeight.w600, fontSize: 15, 
                                                       color: isVerified ? Colors.green[900] : Colors.orange[900]),
                                    ),
                                  ),
                                ],
                              ),
                              if (licenseNumber != 'null' && licenseNumber != 'Unknown' && licenseNumber.isNotEmpty) ...[
                                const SizedBox(height: 6),
                                Padding(
                                  padding: const EdgeInsets.only(left: 30.0),
                                  child: Text("License: $licenseNumber", style: TextStyle(color: Colors.blueGrey[700], fontSize: 13, fontWeight: FontWeight.w500)),
                                ),
                              ],
                              const SizedBox(height: 8),
                              Padding(
                                padding: const EdgeInsets.only(left: 30.0),
                                child: Text(
                                  isVerified ? "Verified against local database" : "Not found in registry",
                                  style: TextStyle(
                                    fontSize: 12, 
                                    fontWeight: FontWeight.w600,
                                    letterSpacing: 0.3,
                                    color: isVerified ? Colors.green[700] : Colors.orange[800]
                                  ),
                                ),
                              ),
                            ],
                          ),
                        )
                      else
                        const Padding(
                          padding: EdgeInsets.only(bottom: 16.0),
                          child: Text("Unknown", style: TextStyle(fontSize: 16)),
                        ),
                      // ---------------------------

                      const Text("Detected Drugs", style: TextStyle(fontWeight: FontWeight.bold, color: Colors.blue)),
                      const SizedBox(height: 4),
                      Text(detectedDrugs, style: const TextStyle(fontSize: 16)),
                      
                      const SizedBox(height: 16),
                      
                      const Text("Full Transcription", style: TextStyle(fontWeight: FontWeight.bold, color: Colors.grey)),
                      const SizedBox(height: 4),
                      Container(
                        width: double.infinity,
                        padding: const EdgeInsets.all(12),
                        decoration: BoxDecoration(
                          color: Colors.grey[50],
                          borderRadius: BorderRadius.circular(8),
                          border: Border.all(color: Colors.grey[200]!),
                        ),
                        child: Text(text, style: const TextStyle(fontFamily: 'RobotoMono', fontSize: 13)),
                      ),
                    ],
                  ),
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    final userId = FirebaseAuth.instance.currentUser?.uid;

    if (userId == null) {
      return Scaffold(
        appBar: AppBar(title: const Text('Prescription History')),
        body: const Center(child: Text('Please log in to see your history.')),
      );
    }

    final prescriptionsStream = FirebaseFirestore.instance
        .collection('users')
        .doc(userId)
        .collection('prescriptions')
        .orderBy('timestamp', descending: true)
        .snapshots();

    return Scaffold(
      appBar: AppBar(title: const Text('Prescription History')),
      body: StreamBuilder<QuerySnapshot>(
        stream: prescriptionsStream,
        builder: (context, snapshot) {
          if (snapshot.connectionState == ConnectionState.waiting) {
            return const Center(child: CircularProgressIndicator());
          }
          if (snapshot.hasError) {
            return Center(child: Text('Error: ${snapshot.error}'));
          }
          if (!snapshot.hasData || snapshot.data!.docs.isEmpty) {
            return Center(
              child: Column(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Icon(Icons.history_edu, size: 100, color: Colors.grey[300]),
                  const SizedBox(height: 20),
                  Text('No Saved Prescriptions', style: TextStyle(fontSize: 22, fontWeight: FontWeight.bold, color: Colors.grey[600])),
                  const SizedBox(height: 10),
                  Text('Your scanned prescriptions will appear here.', textAlign: TextAlign.center, style: TextStyle(fontSize: 16, color: Colors.grey[500])),
                ],
              ),
            );
          }

          final prescriptions = snapshot.data!.docs;

          return ListView.builder(
            padding: const EdgeInsets.all(12.0),
            itemCount: prescriptions.length,
            itemBuilder: (context, index) {
              final doc = prescriptions[index];
              final data = doc.data() as Map<String, dynamic>;
              final detectedDrugs = (data['detectedDrugs'] as List<dynamic>?) ?? [];
              final drugPreview = detectedDrugs.isNotEmpty ? detectedDrugs.first.toString() : 'Prescription Scan';
              final timestamp = data['timestamp'] as Timestamp?;

              return Card(
                margin: const EdgeInsets.symmetric(vertical: 8.0),
                elevation: 2,
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
                child: ListTile(
                  contentPadding: const EdgeInsets.symmetric(vertical: 8.0, horizontal: 16.0),
                  leading: CircleAvatar(
                    backgroundColor: Theme.of(context).primaryColor.withAlpha(26),
                    child: Icon(Icons.receipt_long, color: Theme.of(context).primaryColor),
                  ),
                  title: Text(drugPreview, maxLines: 1, overflow: TextOverflow.ellipsis, style: const TextStyle(fontWeight: FontWeight.w600, fontSize: 16)),
                  subtitle: Padding(
                    padding: const EdgeInsets.only(top: 4.0),
                    child: Text(
                      timestamp != null ? DateFormat.yMMMd().add_jm().format(timestamp.toDate()) : 'No date', 
                      style: TextStyle(color: Colors.grey[600], fontSize: 12)
                    ),
                  ),
                  trailing: Row(
                    mainAxisSize: MainAxisSize.min,
                    children: [
                      IconButton(
                        icon: const Icon(Icons.delete_outline, color: Colors.redAccent),
                        onPressed: () => _deletePrescription(context, doc.id),
                      ),
                      const Icon(Icons.arrow_forward_ios, size: 14, color: Colors.grey),
                    ],
                  ),
                  onTap: () => _viewPrescriptionDetails(context, data),
                ),
              );
            },
          );
        },
      ),
    );
  }
}