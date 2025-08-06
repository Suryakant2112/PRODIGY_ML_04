import cv2

index = 0

def test_camera(index):
    """Tests a camera at a given index."""
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"Error: Could not open camera at index {index}")
        return

    print(f"Success! Camera found at index {index}. Press 'q' to close.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break
        
        cv2.imshow(f"Camera Test (Index {index})", frame)
        
        # q - quit camera
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

# testing some camera indices
for i in range(3):
    test_camera(i)