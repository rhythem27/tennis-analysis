def get_center_of_bbox(bbox):
    x1, y1, x2, y2 = bbox
    centre_x = int((x1 + x2) / 2)
    centre_y = int((y1 + y2) / 2)
    return (centre_x, centre_y)

def measure_distance(p1, p2):
    x1, y1 = p1
    x2, y2 = p2
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5