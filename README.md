# video-edit

My explorations on video editting and computer vision.

## My playing around

- Yolo Model - does object detection but pretty slow
- Hand Rolled Motion Detector via openCV

## Swing alyzer

I love doing kettlebell swings, but then I realized I was doing them wrong. Instead of fixing my swing form, I wrote an app to analyze them.  Independantly, I fixed my swing form. https://youtu.be/BD6Sys1GIuU?si=tVnJMGwbD34SPFw5

[swings.webm](https://github.com/idvorkin/video-edit/assets/280981/023149fc-ac58-44b7-8d72-50a4841f02da)

## Other work

[auto-editor](https://github.com/WyattBlue/auto-editor) - Auto trim w/no motion or audio. Except doesn't quite work

     python3 -m auto_editor ~/downloads/delivery.mp4  --ffmpeg_location `which ffmpeg` --edit motion --motion_threshold 0.05
