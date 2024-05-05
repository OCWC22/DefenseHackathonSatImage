import cv2
import os
from PIL import Image
from time import sleep

# Path to your 'feeds' directory
feeds_path = 'feeds/'

# List of media files
media_files = [
    'image1.jpg', 'image2.jpg', 'image3.jpg', 'image4.jpg', 'image5.jpg',
    'image6.jpg', 'image7.jpg', 'image8.jpg', 'image9.jpg', 'video1.mov'
]

# Display each file
for media in media_files:
    file_path = os.path.join(feeds_path, media)
    
    if media.endswith('.jpg'):
        # Display image
        img = Image.open(file_path)
        img.show()
        sleep(5)  # Display each image for 30 seconds
        img.close()
    elif media.endswith('.mov'):
        # Play video
        cap = cv2.VideoCapture(file_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                cv2.imshow('Video', frame)
                # Press 'q' to quit early or wait 30 milliseconds between frames
                if cv2.waitKey(5) & 0xFF == ord('q'):
                    break
            else:
                break
        cap.release()
        cv2.destroyAllWindows()
        sleep(5)  # Pause 30 seconds after the video before the next media file