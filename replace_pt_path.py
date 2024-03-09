import json

def update_image_path(json_file, old_path, new_path):
    # Read the JSON file
    with open(json_file, 'r') as file:
        data = json.load(file)
    
    # Update 'image_path' in 'train', 'val', and 'test'
    for key in ['train', 'val', 'test']:
        if key in data:
            for item in data[key]:
                item['image_path'] = [path.replace(old_path, new_path) for path in item['image_path']]

    # Write the updated data back to the JSON file
    with open(json_file, 'w') as file:
        json.dump(data, file, indent=4)

# Usage
json_file = 'your_json_file.json'  # Replace with your JSON file path
old_path = '/storage/Pathology/wsi-report/wsi'
new_path = '/new/path/here'        # Replace with your new path
update_image_path(json_file, old_path, new_path)
