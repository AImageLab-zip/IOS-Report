#!/usr/bin/env python3
"""
Test script to verify multi-user support works correctly.
Simulates multiple users and checks they get different patients.
"""

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.label_photos_web import PhotoLabelerWeb

def test_multi_user():
    """Test that multiple users get different patients."""
    
    dataset_root = "/work/grana_maxillo/Dataset_FerraraDump_ToothFairy4M"
    labeler = PhotoLabelerWeb(dataset_root)
    
    print("Testing multi-user support...\n")
    
    # Simulate 3 users
    users = ['user1', 'user2', 'user3']
    assigned_patients = {}
    
    for user in users:
        data = labeler.get_current_patient_data(user)
        if data:
            patient = data['patient_name']
            assigned_patients[user] = patient
            print(f"✓ {user}: assigned patient {patient}")
        else:
            print(f"✗ {user}: no patient available")
    
    # Check all users got different patients
    patient_list = list(assigned_patients.values())
    unique_patients = set(patient_list)
    
    print(f"\nTotal users: {len(users)}")
    print(f"Patients assigned: {len(patient_list)}")
    print(f"Unique patients: {len(unique_patients)}")
    
    if len(patient_list) == len(unique_patients):
        print("\n✅ SUCCESS: All users got different patients!")
        return True
    else:
        print("\n❌ FAILURE: Some users got the same patient!")
        for user, patient in assigned_patients.items():
            count = patient_list.count(patient)
            if count > 1:
                print(f"  Patient {patient} assigned to {count} users")
        return False

if __name__ == "__main__":
    test_multi_user()
