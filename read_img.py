import cv2
import os
import easyocr

def draw_text_boxes(img, results):
    for (bbox, text, prob) in results:
        (top_left, top_right, bottom_right, bottom_left) = bbox
        top_left = tuple(map(int, top_left))
        bottom_right = tuple(map(int, bottom_right))
        cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)
        cv2.putText(img, text, (top_left[0], top_left[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)

def process_images_set(input_folder, output_folder, selected_set):
    reader = easyocr.Reader(['en'])
    image_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    process_ocr_results = globals().get(f'process_ocr_results_set_{selected_set}')
    if process_ocr_results is None:
        print(f"Chưa có xử lý cho bộ khung {selected_set}")
        return
    
    all_keys = {}
    
    for idx, image_file in enumerate(image_files):
        image_path = os.path.join(input_folder, image_file)
        img = cv2.imread(image_path)
        if img is None:
            print(f"Could not read image: {image_path}")
            continue
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        results = reader.readtext(gray)
        draw_text_boxes(img, results)
        # texts_read = [text for (_, text, _) in results]
        filename = os.path.basename(image_path)
        cv2.imwrite(os.path.join(output_folder, filename), img)
        # result_dict = process_ocr_results(idx, texts_read)
        # if result_dict:
            # all_keys.update(result_dict)
    
    print(all_keys)

def process_ocr_results_set_1(ROI_index, texts_read):
    sets = [
        ["Tgian chu kì", "Vtri khuon"],
        ["T gian điền", "V trí EJ", "Đệm tối thiểu"],
        ["T g đ lượng", "V trí t vít", "Áp suất đỉnh điền"]
    ]
    
    if ROI_index < len(sets):
        keys = sets[ROI_index]
        values = texts_read + [" "] * (len(keys) - len(texts_read))
        return dict(zip(keys, values))
    return None

def process_ocr_results_set_2(ROI_index, texts_read):
    sets = [
        ["Vùng 5"],
        ["Vùng 4"],
        ["Vùng 3"]
    ]
    
    if ROI_index < len(sets):
        keys = sets[ROI_index]
        values = texts_read + [" "] * (len(keys) - len(texts_read))
        return dict(zip(keys, values))
    return None
