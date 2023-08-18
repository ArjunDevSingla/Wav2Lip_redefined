import cv2
import os
import subprocess
from datetime import datetime, timedelta
import numpy as np

# Load the pretrained face detection model
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

video_path = 'media/video1.mp4'
audio_path = 'media/output10.wav'

# Initialize Variables
frame_count = 0
face_detected = False
current_audio_timestamp = 0.0
detected_segment_start = None
undetected_segment_start = None
output_frames = []
previous_output_frames = []

# Open Video and Audio
cap = cv2.VideoCapture(video_path)
frame_rate = cap.get(cv2.CAP_PROP_FPS)
audio_cmd = ['ffmpeg', '-i', audio_path, '-f', 's16le', '-ar', '44100', '-ac', '1', '-']

# Create a subprocess to read audio
audio_process = subprocess.Popen(audio_cmd, stdout=subprocess.PIPE, bufsize=10**8)

# Creating an empty audio file
from pydub import AudioSegment

# Create an empty audio segment
empty_audio = AudioSegment.silent(duration=1)  # 1 millisecond

# Add a small amount of silence to the empty audio
empty_audio_with_silence = empty_audio + AudioSegment.silent(duration=1)

# Save the empty audio with silence to a file
empty_audio_with_silence.export('media/output_audio_1.wav', format='wav')
empty_audio_with_silence.export('media/output_audio_2.wav', format='wav')


# def extract_audio_segment_with_face(start_frame, end_frame):
#     # ... (extract audio logic)
#     start_time_stamp = start_frame / frame_rate
#     print(start_time_stamp)
#     end_time_stamp = end_frame / frame_rate
#
#     time_delta_start = timedelta(seconds=start_time_stamp)
#     start_datetime = str(time_delta_start)
#
#     time_delta_end = timedelta(seconds=end_time_stamp)
#     end_datetime = str(time_delta_end)
#
#     output_audio = 'media/output_audio_1.wav'
#
#     extract_audio = ['ffmpeg', '-i', audio_path, '-ss', start_datetime, '-to', end_datetime, '-c:a', 'pcm_s16le', '-strict', 'experimental', '-y', output_audio]
#
#     subprocess.run(extract_audio)
#
# def extract_audio_segment_without_face(start_frame, end_frame):
#     # ... (extract audio logic)
#     start_time_stamp = start_frame / frame_rate
#     print(start_time_stamp)
#     end_time_stamp = end_frame / frame_rate
#
#     time_delta_start = timedelta(seconds=start_time_stamp)
#     start_datetime = str(time_delta_start)
#
#     time_delta_end = timedelta(seconds=end_time_stamp)
#     end_datetime = str(time_delta_end)
#
#     output_audio = 'media/output_audio_2.wav'
#
#     extract_audio = ['ffmpeg', '-i', audio_path, '-ss', start_datetime, '-to', end_datetime, '-c:a', 'pcm_s16le', '-strict', 'experimental', '-y', output_audio]
#
#     subprocess.run(extract_audio)

def extract_audio_segment_with_face(start_frame, end_frame, segment_index):
    # ... (extract audio logic)
    start_time_stamp = start_frame / frame_rate
    end_time_stamp = end_frame / frame_rate

    time_delta_start = timedelta(seconds=start_time_stamp)
    start_datetime = str(time_delta_start)

    time_delta_end = timedelta(seconds=end_time_stamp)
    end_datetime = str(time_delta_end)

    output_audio = 'media/output_audio_with_face_{}.wav'.format(segment_index)  # Use unique filename with segment index

    extract_audio = ['ffmpeg', '-i', audio_path, '-ss', start_datetime, '-to', end_datetime, '-c:a', 'pcm_s16le', '-strict', 'experimental', '-y', output_audio]

    subprocess.run(extract_audio)

def extract_audio_segment_without_face(start_frame, end_frame, segment_index):
    # ... (extract audio logic)
    start_time_stamp = start_frame / frame_rate
    end_time_stamp = end_frame / frame_rate

    time_delta_start = timedelta(seconds=start_time_stamp)
    start_datetime = str(time_delta_start)

    time_delta_end = timedelta(seconds=end_time_stamp)
    end_datetime = str(time_delta_end)

    output_audio = 'media/output_audio_without_face_{}.wav'.format(segment_index)  # Use unique filename with segment index

    extract_audio = ['ffmpeg', '-i', audio_path, '-ss', start_datetime, '-to', end_datetime, '-c:a', 'pcm_s16le', '-strict', 'experimental', '-y', output_audio]

    subprocess.run(extract_audio)

def get_frame_char(video_path):

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Get the frame width and frame height
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Release the video capture object
    cap.release()

    return frame_width, frame_height, fps

# Making an empty video
frame_width, frame_height, fps = get_frame_char(video_path)
duration = 1

# Create a VideoWriter object to write the video
# Create a VideoWriter object to write the video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('media/temp_video_1.mp4', fourcc, fps, (frame_width, frame_height))
out_1 = cv2.VideoWriter('media/temp_video.mp4', fourcc, fps, (frame_width, frame_height))

# Create an empty black frame
empty_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

# Write the empty frame multiple times to create the video
for _ in range(int(fps * duration)):
    out.write(empty_frame)

for _ in range(int(fps * duration)):
    out_1.write(empty_frame)

# Release the VideoWriter
out.release()
out_1.release()


def save_segments_to_files_with_face(video_path, start_frame, end_frame, audio_segment):
    # ... (other logic)
    temp_video_path = 'media/temp_video_1.mp4'
    outfile_path = 'media/results_voice_{}.mp4'.format(start_frame)

    # Video parameters
    frame_width, frame_height, fps = get_frame_char(video_path)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_video_path, fourcc, fps, (frame_width, frame_height))

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)  # Set starting frame

    for _ in range(start_frame, end_frame + 1):
        ret, frame = cap.read()
        if ret:
            out.write(frame)

    cap.release()
    out.release()

    command = "python inference.py --checkpoint_path checkpoints/wav2lip_gan.pth --face {} --audio {} --nosmooth --pads 20 0 0 0 --outfile {}".format(
        temp_video_path, audio_segment, outfile_path)

    subprocess.run(command, shell=True)

    # Clean up temporary files


def save_segments_to_files_without_face(video_path, start_frame, end_frame, audio_segment):
    # ... (other logic)
    temp_video_path = 'media/temp_video.mp4'

    # Video parameters
    frame_width, frame_height, fps = get_frame_char(video_path)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_video_path, fourcc, fps, (frame_width, frame_height))

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)  # Set starting frame

    for _ in range(start_frame, end_frame + 1):
        ret, frame = cap.read()
        if ret:
            out.write(frame)

    cap.release()
    out.release()

    output_video = 'media/empty_video_no_face_{}.mp4'.format(start_frame)  # Update the filename

    # Merge audio and video
    command = 'ffmpeg -i "{}" -i "{}" -c:v copy -c:a aac -strict experimental -y {}'.format(
        temp_video_path, audio_segment, output_video
    )

    subprocess.run(command, shell=True)

    # Clean up temporary files


while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Convert the frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect Faces in the frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=15, minSize=(30, 30))

    if len(faces) > 0:
        # Face detected
        if not face_detected:
            face_detected = True
            audio_process.kill()
            current_audio_timestamp = frame_count / frame_rate
            audio_cmd = ['ffmpeg', '-ss', str(current_audio_timestamp), '-i', audio_path, '-f', 's16le', '-ar', '44100',
                         '-ac', '1', '-']
            audio_process = subprocess.Popen(audio_cmd, stdout=subprocess.PIPE, bufsize=10 ** 8)
            detected_segment_start = frame_count  # Update detected_segment_start
            if undetected_segment_start is not None:
                # Save the previous undetected segment if exists
                previous_output_frames.append((undetected_segment_start, frame_count - 1))
            undetected_segment_start = None  # Reset undetected_segment_start
    else:
        # No face detected
        if face_detected:
            face_detected = False
            audio_process.kill()
            audio_cmd = ['ffmpeg', '-ss', str(current_audio_timestamp), '-i', audio_path, '-f', 's16le', '-ar', '44100',
                         '-ac', '1', '-']
            audio_process = subprocess.Popen(audio_cmd, stdout=subprocess.PIPE, bufsize=10 ** 8)
            undetected_segment_start = frame_count  # Update undetected_segment_start
            if detected_segment_start is not None:
                # Save the previous detected segment if exists
                output_frames.append((detected_segment_start, frame_count - 1))
            detected_segment_start = None  # Reset detected_segment_start

    frame_count += 1

    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Display the frame with face detection
    cv2.imshow('Face Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Save the last detected or undetected segment
if detected_segment_start is not None:
    output_frames.append((detected_segment_start, frame_count - 1))
if undetected_segment_start is not None:
    previous_output_frames.append((undetected_segment_start, frame_count - 1))

print(output_frames)

for index, (start_frame, end_frame) in enumerate(output_frames):
    extract_audio_segment_with_face(start_frame, end_frame, index)
    save_segments_to_files_with_face(video_path, start_frame, end_frame, 'media/output_audio_with_face_{}.wav'.format(index))

for index, (start_frame, end_frame) in enumerate(previous_output_frames):
    extract_audio_segment_without_face(start_frame, end_frame, index)
    save_segments_to_files_without_face(video_path, start_frame, end_frame, 'media/output_audio_without_face_{}.wav'.format(index))


cap.release()
audio_process.kill()
cv2.destroyAllWindows()
