from src.dataset.ferrara_dump_dataset import FerraraDumpDataset
from tqdm import tqdm

    
def compute_average_image_resolution(dataset):
    total_width = 0
    total_height = 0
    total_images = 0
    num_patients = len(dataset)

    for idx in tqdm(range(num_patients)):
        patient = dataset[idx]
        for image in patient['images']:
            total_images += 1
            total_width += image.shape[2]
            total_height += image.shape[1]

    avg_width = total_width / total_images
    avg_height = total_height / total_images

    return avg_width, avg_height

if __name__ == "__main__":
    import time

    
    start_time = time.time()
    dataset = FerraraDumpDataset(
        root_dir="/work/grana_maxillo/Dataset_FerraraDump_ToothFairy4M/",
        preload=False,
        load_meshes=False,
        load_images=True,
        cache_in_ram=False,
        num_workers=8,
    )
    load_time = time.time() - start_time
    
   
    avg_width, avg_height = compute_average_image_resolution(dataset)
    print(f"\nAverage image resolution: {avg_width:.1f}x{avg_height:.1f}")
