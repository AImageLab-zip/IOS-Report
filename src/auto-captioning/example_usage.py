#!/usr/bin/env python3
"""
Example script showing how to use the generated descriptions programmatically.
"""

import json
from pathlib import Path
from typing import Dict, List


class DescriptionLoader:
    """Load and work with generated descriptions."""
    
    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = Path(output_dir)
    
    def load_patient(self, patient_id: str) -> Dict:
        """Load description for a single patient."""
        json_file = self.output_dir / f"{patient_id}.json"
        
        if not json_file.exists():
            raise FileNotFoundError(f"Patient {patient_id} not found")
        
        with open(json_file, 'r') as f:
            return json.load(f)
    
    def load_all(self) -> List[Dict]:
        """Load all patient descriptions."""
        all_file = self.output_dir / "all_descriptions.json"
        
        if all_file.exists():
            with open(all_file, 'r') as f:
                return json.load(f)
        
        # Fallback: load individual files
        results = []
        for json_file in self.output_dir.glob("*.json"):
            if json_file.name == "all_descriptions.json":
                continue
            with open(json_file, 'r') as f:
                results.append(json.load(f))
        
        return sorted(results, key=lambda x: x['patient_id'])
    
    def parse_description(self, description: str) -> Dict:
        """
        Parse the structured description into sections.
        Returns a dictionary with each section as a key.
        """
        sections = {
            'transverse': [],
            'vertical': [],
            'sagittal': [],
            'midline': [],
            'crowding': [],
            'arch_form': [],
            'additional': []
        }
        
        current_section = None
        
        for line in description.split('\n'):
            line = line.strip()
            
            if not line or line.startswith('Patient Intra-Oral Description'):
                continue
            
            # Detect section headers
            if 'transverse' in line.lower():
                current_section = 'transverse'
            elif 'vertical' in line.lower():
                current_section = 'vertical'
            elif 'sagittal' in line.lower():
                current_section = 'sagittal'
            elif 'midline' in line.lower():
                current_section = 'midline'
            elif 'crowding' in line.lower() or 'spacing' in line.lower():
                current_section = 'crowding'
            elif 'arch form' in line.lower():
                current_section = 'arch_form'
            elif 'additional' in line.lower() or 'observations' in line.lower():
                current_section = 'additional'
            elif current_section and line.startswith(('-', '•', '*')):
                sections[current_section].append(line[1:].strip())
            elif current_section and line and not line[0].isdigit():
                sections[current_section].append(line)
        
        return sections


# Example usage functions

def example_1_load_patient():
    """Example: Load and display a single patient."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Load Single Patient")
    print("="*60)
    
    loader = DescriptionLoader()
    
    try:
        patient = loader.load_patient("201_1979")
        print(f"\nPatient ID: {patient['patient_id']}")
        print(f"Model: {patient['model']}")
        print(f"Timestamp: {patient['timestamp']}")
        print(f"\nDescription:\n{patient['description'][:500]}...")
    except FileNotFoundError as e:
        print(f"Error: {e}")


def example_2_load_all():
    """Example: Load all patients and show summary."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Load All Patients")
    print("="*60)
    
    loader = DescriptionLoader()
    patients = loader.load_all()
    
    print(f"\nTotal patients: {len(patients)}")
    print(f"\nFirst 5 patients:")
    for patient in patients[:5]:
        desc_len = len(patient['description'])
        print(f"  - {patient['patient_id']}: {desc_len} characters")


def example_3_parse_sections():
    """Example: Parse description into sections."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Parse Description Sections")
    print("="*60)
    
    loader = DescriptionLoader()
    
    try:
        patient = loader.load_patient("201_1979")
        sections = loader.parse_description(patient['description'])
        
        print(f"\nPatient: {patient['patient_id']}")
        print("\nParsed sections:")
        for section, content in sections.items():
            if content:
                print(f"\n{section.upper()}:")
                for line in content[:3]:  # Show first 3 lines
                    print(f"  - {line}")
    except FileNotFoundError as e:
        print(f"Error: {e}")


def example_4_filter_by_condition():
    """Example: Filter patients by specific condition."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Filter by Condition")
    print("="*60)
    
    loader = DescriptionLoader()
    patients = loader.load_all()
    
    # Find patients with Class II malocclusion
    condition = "Class II"
    matching = [p for p in patients if condition in p['description']]
    
    print(f"\nSearching for: '{condition}'")
    print(f"Found {len(matching)} patients out of {len(patients)}")
    
    if matching:
        print(f"\nFirst 3 matches:")
        for patient in matching[:3]:
            print(f"  - {patient['patient_id']}")


def example_5_statistics():
    """Example: Compute statistics."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Compute Statistics")
    print("="*60)
    
    loader = DescriptionLoader()
    patients = loader.load_all()
    
    if not patients:
        print("No patients found")
        return
    
    # Description lengths
    lengths = [len(p['description']) for p in patients]
    
    print(f"\nDescription Statistics:")
    print(f"  Total patients: {len(patients)}")
    print(f"  Average length: {sum(lengths)/len(lengths):.0f} characters")
    print(f"  Min length: {min(lengths)} characters")
    print(f"  Max length: {max(lengths)} characters")
    
    # Common terms
    keywords = ["Class I", "Class II", "Class III", "crossbite", "crowding", "deep overbite"]
    print(f"\nKeyword frequency:")
    for keyword in keywords:
        count = sum(1 for p in patients if keyword.lower() in p['description'].lower())
        percentage = (count / len(patients)) * 100
        print(f"  {keyword}: {count} ({percentage:.1f}%)")


def main():
    """Run all examples."""
    print("\n" + "="*60)
    print("DESCRIPTION USAGE EXAMPLES")
    print("="*60)
    
    examples = [
        example_1_load_patient,
        example_2_load_all,
        example_3_parse_sections,
        example_4_filter_by_condition,
        example_5_statistics
    ]
    
    for example in examples:
        try:
            example()
        except Exception as e:
            print(f"\nError in {example.__name__}: {e}")
    
    print("\n" + "="*60)
    print("Examples completed!")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
