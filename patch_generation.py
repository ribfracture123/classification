import os
import json
import numpy as np
import nibabel as nib
import pandas as pd
from pathlib import Path
from collections import Counter

def load_json_file(json_path):
    """Load a single JSON file."""
    with open(json_path, 'r') as f:
        return json.load(f)

def get_matching_files(case_id, ct_folder, label_folder, json_folder):
    """Get matching files for a given case ID."""
    ct_file = os.path.join(ct_folder, f"{case_id}-image.nii")
    label_file = os.path.join(label_folder, f"{case_id}-label.nii")
    json_file = os.path.join(json_folder, f"{case_id}-image.json")
    
    if not all(os.path.exists(f) for f in [ct_file, label_file, json_file]):
        return None
    return ct_file, label_file, json_file

def normalize_case_id(filename):
    """Extract and normalize case ID from filename."""
    return filename.replace('-image', '').replace('-label', '').replace('.nii', '').replace('.json', '')

def calculate_box_volume(box):
    """Calculate volume of a 3D bounding box."""
    width = box[3] - box[0]
    height = box[4] - box[1]
    depth = box[5] - box[2]
    return width * height * depth

def get_most_frequent_label(label_patch):
    """
    Get the most frequently occurring non-zero label value.
    Returns tuple of (label_value, frequency, total_pixels)
    """
    # Get all non-zero labels and their counts
    labels = label_patch[label_patch > 0]
    if len(labels) == 0:
        return 0, 0, 0
    
    # Count occurrences of each label
    label_counts = Counter(labels)
    
    # Find the label with maximum frequency
    most_common_label, frequency = label_counts.most_common(1)[0]
    total_pixels = len(labels)
    
    return int(most_common_label), frequency, total_pixels

def aggregate_boxes(json_data):
    """Aggregate 2D boxes into 3D bounding boxes with improved tracking."""
    aggregated_boxes = []
    current_boxes = []
    current_start_idx = []
    last_img_idx = -1

    sorted_data = sorted(json_data['data'], key=lambda x: x['img_idx'])
    
    for entry in sorted_data:
        img_idx = entry['img_idx']
        boxes = entry.get('boxes', [])
        
        # Handle sequence gaps
        if img_idx != last_img_idx + 1 and current_boxes:
            for i, current_box in enumerate(current_boxes):
                if current_start_idx[i] < last_img_idx:
                    aggregated_box = current_box[:2] + [current_start_idx[i]] + current_box[2:] + [last_img_idx]
                    if calculate_box_volume(aggregated_box) > 0:
                        aggregated_boxes.append(aggregated_box)
            current_boxes = []
            current_start_idx = []

        if not boxes:
            last_img_idx = img_idx
            continue

        if not current_boxes:
            current_boxes = boxes
            current_start_idx = [img_idx] * len(boxes)
        else:
            matched_indices = set()
            new_current_boxes = []
            new_current_start_idx = []
            
            # Match boxes between frames
            for i, current_box in enumerate(current_boxes):
                best_match = None
                best_match_score = float('inf')
                
                current_center = np.array([(current_box[0] + current_box[2])/2, 
                                         (current_box[1] + current_box[3])/2])
                current_size = (current_box[2] - current_box[0]) * (current_box[3] - current_box[1])
                
                for j, new_box in enumerate(boxes):
                    if j in matched_indices:
                        continue
                        
                    new_center = np.array([(new_box[0] + new_box[2])/2, 
                                         (new_box[1] + new_box[3])/2])
                    new_size = (new_box[2] - new_box[0]) * (new_box[3] - new_box[1])
                    
                    dist = np.linalg.norm(current_center - new_center)
                    size_ratio = min(current_size, new_size) / max(current_size, new_size)
                    
                    match_score = dist * (1 / size_ratio)
                    
                    if dist < 20 and size_ratio > 0.5 and match_score < best_match_score:
                        best_match = j
                        best_match_score = match_score
                
                if best_match is not None:
                    matched_indices.add(best_match)
                    matched_box = boxes[best_match]
                    averaged_box = [
                        (current_box[0] + matched_box[0])/2,
                        (current_box[1] + matched_box[1])/2,
                        (current_box[2] + matched_box[2])/2,
                        (current_box[3] + matched_box[3])/2
                    ]
                    new_current_boxes.append(averaged_box)
                    new_current_start_idx.append(current_start_idx[i])
                else:
                    if last_img_idx - current_start_idx[i] >= 2:
                        aggregated_box = current_box[:2] + [current_start_idx[i]] + current_box[2:] + [last_img_idx]
                        if calculate_box_volume(aggregated_box) > 0:
                            aggregated_boxes.append(aggregated_box)
            
            for j, new_box in enumerate(boxes):
                if j not in matched_indices:
                    new_current_boxes.append(new_box)
                    new_current_start_idx.append(img_idx)
            
            current_boxes = new_current_boxes
            current_start_idx = new_current_start_idx
        
        last_img_idx = img_idx

    # Handle remaining boxes
    for i, current_box in enumerate(current_boxes):
        if last_img_idx - current_start_idx[i] >= 2:
            aggregated_box = current_box[:2] + [current_start_idx[i]] + current_box[2:] + [last_img_idx]
            if calculate_box_volume(aggregated_box) > 0:
                aggregated_boxes.append(aggregated_box)
    
    return aggregated_boxes

def extract_patch(image, center, patch_size=(64, 64, 32), padding_value=0):
    """
    Extract a patch from the image centered at the given coordinates with proper padding.
    """
    image_data = image.get_fdata()
    x, y, z = center
    half_x, half_y, half_z = patch_size[0]//2, patch_size[1]//2, patch_size[2]//2
    
    # Calculate patch boundaries
    x_start = max(0, x - half_x)
    x_end = min(image_data.shape[0], x + half_x)
    y_start = max(0, y - half_y)
    y_end = min(image_data.shape[1], y + half_y)
    z_start = max(0, z - half_z)
    z_end = min(image_data.shape[2], z + half_z)
    
    # Create padded patch
    patch = np.full(patch_size, padding_value, dtype=image_data.dtype)
    
    # Calculate patch indices
    patch_x_start = half_x - (x - x_start)
    patch_y_start = half_y - (y - y_start)
    patch_z_start = half_z - (z - z_start)
    
    patch_x_end = patch_x_start + (x_end - x_start)
    patch_y_end = patch_y_start + (y_end - y_start)
    patch_z_end = patch_z_start + (z_end - z_start)
    
    # Copy data to patch
    patch[patch_x_start:patch_x_end,
          patch_y_start:patch_y_end,
          patch_z_start:patch_z_end] = image_data[x_start:x_end,
                                                 y_start:y_end,
                                                 z_start:z_end]
    
    return patch

def process_ct_scans(ct_folder, label_folder, json_folder, output_folder):
    """Process CT scans and extract patches with identity affine transformation."""
    # Create output directories
    output_folder = Path(output_folder)
    ct_patches_dir = output_folder / "ct_patches"
    label_patches_dir = output_folder / "label_patches"
    
    for dir_path in [output_folder, ct_patches_dir, label_patches_dir]:
        dir_path.mkdir(exist_ok=True, parents=True)
    
    # Get unique case IDs
    case_ids = set()
    for filename in os.listdir(ct_folder):
        if filename.endswith('-image.nii'):
            case_ids.add(normalize_case_id(filename))
    
    patch_data = []
    total_cases = len(case_ids)
    
    # Create identity affine matrix
    identity_affine = np.eye(4)
    
    for idx, case_id in enumerate(sorted(case_ids), 1):
        print(f"Processing case {case_id} ({idx}/{total_cases})...")
        
        # Get matching files
        file_set = get_matching_files(case_id, ct_folder, label_folder, json_folder)
        if not file_set:
            print(f"Skipping case {case_id} - missing files")
            continue
            
        ct_file, label_file, json_file = file_set
        
        try:
            # Load data
            ct_image = nib.load(ct_file)
            label_image = nib.load(label_file)
            json_data = load_json_file(json_file)
            
            # Process boxes
            boxes = aggregate_boxes(json_data)
            
            for box_idx, box in enumerate(boxes):
                center = (
                    int((box[1] + box[4]) / 2),
                    int((box[0] + box[3]) / 2),
                    int((box[2] + box[5]) / 2)
                )
                
                # Extract patches
                ct_patch_data = extract_patch(ct_image, center)
                label_patch_data = extract_patch(label_image, center)
                
                # Get most frequent non-zero label and its statistics
                most_common_label, frequency, total_pixels = get_most_frequent_label(label_patch_data)
                has_fracture = int(most_common_label > 0)
                
                # Create patch name based on presence of fracture
                if has_fracture:
                    patch_name = f"{case_id}_patch_{box_idx}_{most_common_label}"
                else:
                    patch_name = f"{case_id}_patch_{box_idx}"
                
                # Create and save NIfTI images with identity affine
                ct_patch_nii = nib.Nifti1Image(ct_patch_data, identity_affine)
                label_patch_nii = nib.Nifti1Image(label_patch_data, identity_affine)
                
                nib.save(ct_patch_nii, str(ct_patches_dir / f"{patch_name}_ct.nii.gz"))
                nib.save(label_patch_nii, str(label_patches_dir / f"{patch_name}_label.nii.gz"))
                
                # Store metadata
                patch_data.append({
                    'patch_name': patch_name,
                    'AIRib ID': case_id,
                    'class_label': has_fracture,
                    'Label': most_common_label,
                    'label_frequency': frequency,
                    'total_label_pixels': total_pixels,
                    'center_x': center[0],
                    'center_y': center[1],
                    'center_z': center[2],
                    'box_start_slice': int(box[2]),
                    'box_end_slice': int(box[5])
                   
                })
                
        except Exception as e:
            print(f"Error processing case {case_id}: {str(e)}")
            continue
    
    # Save metadata
    df = pd.DataFrame(patch_data)
    df.to_csv(output_folder / 'patch_metadata.csv', index=False)
    
    # Print summary
    print("\nProcessing Summary:")
    print(f"Total cases processed: {total_cases}")
    print(f"Total patches extracted: {len(patch_data)}")
    print(f"Patches with fractures: {sum(1 for x in patch_data if x['class_label'] == 1)}")
    print(f"Patches without fractures: {sum(1 for x in patch_data if x['class_label'] == 0)}")
    
    # Save summary to file
    with open(output_folder / 'processing_summary.txt', 'w') as f:
        f.write(f"Processing Summary:\n")
        f.write(f"Total cases processed: {total_cases}\n")
        f.write(f"Total patches extracted: {len(patch_data)}\n")
        f.write(f"Patches with fractures: {sum(1 for x in patch_data if x['class_label'] == 1)}\n")
        f.write(f"Patches without fractures: {sum(1 for x in patch_data if x['class_label'] == 0)}\n")

def main():
    # Define input and output paths
    ct_folder = "/workspace/ribfrac/Data/Test/images"
    label_folder = "/workspace/ribfrac/Data/Test/labels"
    json_folder = "./check_postprocessing_airrib"
    output_folder = "./patches_airrib"
    
    # Print directory contents for verification
    print("\nVerifying directory contents...")
    for folder, name in [(ct_folder, "CT Images"), (label_folder, "Labels"), (json_folder, "JSON")]:
        print(f"\n{name} directory contents:")
        try:
            print(sorted(os.listdir(folder)))
        except Exception as e:
            print(f"Error accessing {folder}: {str(e)}")
    
    print("\nStarting processing...")
    process_ct_scans(ct_folder, label_folder, json_folder, output_folder)

if __name__ == "__main__":
    main()