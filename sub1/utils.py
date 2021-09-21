def img2feature(object_visual, feature_extractor):
    image_features = feature_extractor(images=object_visual, return_tensors="pt")['pixel_values']
    
    return image_features