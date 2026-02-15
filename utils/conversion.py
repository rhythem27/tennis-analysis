def convert_pixel_distance_to_meters(pixel_distance, refernce_height_in_meters, refernce_height_in_pixels ):
    return (pixel_distance * refernce_height_in_meters) / refernce_height_in_pixels

def convert_meters_to_pixel_distance(meters, refernce_height_in_meters, refernce_height_in_pixels ):
    return (meters * refernce_height_in_pixels) / refernce_height_in_meters
    