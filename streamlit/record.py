import cv2

# Initialize the video capture object
cap = cv2.VideoCapture(0)

# Set the frame rate and the video length
fps = 30
length = 3
num_frames = 180

# Set the width and height of the frames
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Initialize the FourCC codec and create the video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('video.mp4', fourcc, fps, (width, height))

# Create a window to display the video
cv2.namedWindow("Video", cv2.WINDOW_NORMAL)

# Capture and save the frames
for i in range(num_frames):
    ret, frame = cap.read()
    if ret:
        # Flip the frame horizontally
        frame = cv2.flip(frame, 1)
        if i >=0 and i < 31:
            cv2.putText(frame, "3", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            out.write(frame)
        if i >=31 and i < 61:
            cv2.putText(frame, "2", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            out.write(frame)
        if i >=61 and i < 91:
            cv2.putText(frame, "1", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            out.write(frame)
        # Write the frame to the video file if the index is greater than or equal to 91
        if i >= 91:
            cv2.putText(frame, "Recording started!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            out.write(frame)
        # Display the frame in the window
        cv2.imshow("Video", frame)
        cv2.waitKey(1)

# Release the video capture and writer objects
cap.release()
out.release()
